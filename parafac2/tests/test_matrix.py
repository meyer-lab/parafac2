"""Tests for the duck-typed matrix contract described in ``parafac2.matrix``.

A matrix type beyond ``np.ndarray``/``scipy.sparse`` only has to implement
``.shape``, ``.dtype``, ``__matmul__``/``__rmatmul__``, ``norm_sq()``,
``slice_norms(condition_idxs, n_cond)``, and (to run on a GPU backend)
``to_device(backend)``, to be usable as ``X`` throughout the PARAFAC2 fit.
These tests exercise that contract directly with a minimal fake, standing in
for a real implementation such as ``vsparse``'s normalized views, and check
that the built-in wrappers satisfy the same contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_array

from ..matrix import (
    CSRMatrix,
    DenseMatrix,
    Matrix,
    as_linear_operator,
    as_matrix,
    to_gpu,
)
from ..utils import calc_W, randomized_svd_right


class _FakeCenteredMatrix:
    """A duck-typed matrix that centers itself internally (like a vsparse
    normalized view), so it never needs an external `means` argument."""

    # Required so that `ndarray @ fake` defers to this class's own
    # __rmatmul__ instead of numpy trying (and failing) to broadcast it as a
    # 0-d object array -- see the contract in `parafac2.matrix`.
    __array_ufunc__ = None

    def __init__(self, dense: np.ndarray) -> None:
        self._dense = np.asarray(dense, dtype=np.float64)

    @property
    def shape(self) -> tuple[int, int]:
        return self._dense.shape

    @property
    def dtype(self) -> np.dtype:
        return self._dense.dtype

    def __matmul__(self, rhs):
        return self._dense @ rhs

    def __rmatmul__(self, lhs):
        return lhs @ self._dense

    def norm_sq(self) -> float:
        return float(np.sum(self._dense**2))

    def slice_norms(self, condition_idxs, n_cond) -> np.ndarray:
        idxs = np.asarray(condition_idxs)
        return np.array([np.linalg.norm(self._dense[idxs == i]) for i in range(n_cond)])


class _FakeDeviceCapableMatrix(_FakeCenteredMatrix):
    """Like `_FakeCenteredMatrix`, but declares a device-transfer hook."""

    def __init__(self, dense: np.ndarray) -> None:
        super().__init__(dense)
        self.to_device_calls: list[str] = []

    def to_device(self, backend: str) -> _FakeCenteredMatrix:
        self.to_device_calls.append(backend)
        return _FakeCenteredMatrix(self._dense)


@pytest.fixture
def dense() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(6, 4))


# ---------------------------------------------------------------------------
# as_matrix: the one place that looks at the data's type
# ---------------------------------------------------------------------------


def test_as_matrix_wraps_numpy_and_scipy(dense):
    assert isinstance(as_matrix(dense), DenseMatrix)
    assert isinstance(as_matrix(csr_array(dense)), CSRMatrix)


def test_as_matrix_passes_a_duck_typed_matrix_through_untouched(dense):
    fake = _FakeCenteredMatrix(dense)
    assert as_matrix(fake) is fake


def test_as_matrix_rejects_external_means_alongside_a_self_centering_type(dense):
    fake = _FakeCenteredMatrix(dense)
    with pytest.raises(ValueError, match="its own centering"):
        as_matrix(fake, means=np.ones(dense.shape[1]))


def test_as_matrix_allows_all_zero_means_alongside_a_self_centering_type(dense):
    fake = _FakeCenteredMatrix(dense)
    assert as_matrix(fake, means=np.zeros(dense.shape[1])) is fake


# ---------------------------------------------------------------------------
# The numerical routines reach the data only through the contract
# ---------------------------------------------------------------------------


def test_norms_dispatch_to_the_duck_typed_methods(dense):
    fake = _FakeCenteredMatrix(dense)
    assert fake.norm_sq() == pytest.approx(np.sum(dense**2))

    idxs = np.array([0, 0, 1, 1, 1, 2])
    expected = np.array([np.linalg.norm(dense[idxs == i]) for i in range(3)])
    np.testing.assert_allclose(fake.slice_norms(idxs, 3), expected)


def test_calc_W_works_on_a_duck_typed_matrix(dense):
    fake = _FakeCenteredMatrix(dense)
    C = np.ones((dense.shape[1], 2))
    np.testing.assert_allclose(calc_W(fake, C), dense @ C)


def test_randomized_svd_right_works_on_a_duck_typed_matrix(dense):
    """It only ever touches the data via `X @ rhs` / `lhs @ X`."""
    fake = _FakeCenteredMatrix(dense)
    Q = randomized_svd_right(fake, n_components=3, random_state=0)
    assert Q.shape == (dense.shape[1], 3)
    np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=1e-8)


# ---------------------------------------------------------------------------
# as_linear_operator: the adapter SciPy's decompositions see
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wrap", [lambda d: d, csr_array, _FakeCenteredMatrix])
def test_as_linear_operator_matches_the_dense_products(dense, wrap):
    """Every product SciPy can ask for must agree with the dense matrix."""
    op = as_linear_operator(wrap(dense))
    rng = np.random.default_rng(1)
    n_rows, n_cols = dense.shape
    v, u = rng.normal(size=n_cols), rng.normal(size=n_rows)
    V, U = rng.normal(size=(n_cols, 3)), rng.normal(size=(n_rows, 3))

    assert op.shape == dense.shape
    assert op.dtype == np.float64
    np.testing.assert_allclose(op.matvec(v), dense @ v)
    np.testing.assert_allclose(op.rmatvec(u), u @ dense)
    np.testing.assert_allclose(op.matmat(V), dense @ V)
    np.testing.assert_allclose(op.rmatmat(U), dense.T @ U)


def test_as_linear_operator_applies_the_matrix_centering(dense):
    """`means` reaches the operator through the wrapped matrix, not around it."""
    means = np.arange(dense.shape[1], dtype=np.float64)
    op = as_linear_operator(csr_array(dense), means)
    v = np.ones(dense.shape[1])
    np.testing.assert_allclose(op.matvec(v), (dense - means) @ v)


def test_as_linear_operator_is_float64_for_float32_data():
    """A float32 matrix still yields a float64 operator: SciPy's
    decompositions need double precision, while the product itself stays in
    the data's own dtype so the data is never upcast."""
    mat = np.arange(12, dtype=np.float32).reshape(4, 3)
    op = as_linear_operator(mat)
    assert op.dtype == np.float64
    assert op.matvec(np.ones(3)).dtype == np.float64


# ---------------------------------------------------------------------------
# to_gpu dispatch
# ---------------------------------------------------------------------------


def test_to_gpu_passes_through_a_custom_type_on_cpu(dense):
    fake = _FakeCenteredMatrix(dense)
    assert to_gpu(fake, backend="cpu") is fake


def test_to_gpu_uses_to_device_for_a_gpu_backend(dense):
    fake = _FakeDeviceCapableMatrix(dense)
    gpu_mat = to_gpu(fake, backend="cupy")

    assert fake.to_device_calls == ["cupy"]
    assert gpu_mat is not fake  # the object to_device() returned

    rhs = np.ones((dense.shape[1], 2))
    np.testing.assert_allclose(gpu_mat @ rhs, dense @ rhs)


def test_to_gpu_raises_a_clear_error_without_to_device(dense):
    fake = _FakeCenteredMatrix(dense)
    with pytest.raises(TypeError, match="to_device"):
        to_gpu(fake, backend="cupy")


# ---------------------------------------------------------------------------
# The built-in wrappers satisfy the same contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sparse", [False, True])
def test_wrappers_center_implicitly(dense, sparse):
    """`X @ rhs` and `lhs @ X` must already account for the means."""
    means = np.arange(dense.shape[1], dtype=np.float64)
    raw = csr_array(dense) if sparse else dense
    X = as_matrix(raw, means)
    centered = dense - means

    assert isinstance(X, Matrix)
    assert X.shape == dense.shape

    for rhs in (np.ones((dense.shape[1], 3)), np.ones(dense.shape[1])):
        np.testing.assert_allclose(X @ rhs, centered @ rhs)
    for lhs in (np.ones((2, dense.shape[0])), np.ones(dense.shape[0])):
        np.testing.assert_allclose(lhs @ X, lhs @ centered)

    assert X.norm_sq() == pytest.approx(np.sum(centered**2))

    idxs = np.array([0, 0, 0, 1, 1, 1])
    expected = np.array([np.linalg.norm(centered[idxs == i]) for i in range(2)])
    np.testing.assert_allclose(X.slice_norms(idxs, 2), expected)


@pytest.mark.parametrize("sparse", [False, True])
def test_wrapper_products_are_float64_whatever_the_data_dtype(dense, sparse):
    """Callers downstream assume a float64 result even for float32 data."""
    mat32 = dense.astype(np.float32)
    X = as_matrix(csr_array(mat32) if sparse else mat32)
    assert X.dtype == np.float32
    assert (X @ np.ones((dense.shape[1], 2))).dtype == np.float64
    assert (np.ones((2, dense.shape[0])) @ X).dtype == np.float64


def test_all_zero_means_is_treated_as_no_centering(dense):
    assert as_matrix(dense, np.zeros(dense.shape[1])).means is None
