"""Tests for the duck-typed matrix contract described in ``parafac2.utils``.

A matrix type beyond ``np.ndarray``/``scipy.sparse`` only has to implement
``.shape``, ``.dtype``, ``__matmul__``/``__rmatmul__``, and optionally
``norm_sq()``/``slice_norms(condition_idxs, n_cond)`` and
``to_device(backend)``, to be usable as ``X`` throughout the PARAFAC2 fit.
These tests exercise that contract directly with a minimal fake, standing in
for a real implementation such as ``vsparse``'s normalized views.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..backend import GPUMatrix
from ..utils import calc_norm_sq, calc_slice_norms


class _FakeCenteredMatrix:
    """A duck-typed matrix that centers itself internally (like a vsparse
    normalized view), so it never needs an external `means` argument."""

    # Required so that `ndarray @ fake`/`fake_is_rhs` defers to this class's
    # own __rmatmul__ instead of numpy trying (and failing) to broadcast it
    # as a 0-d object array -- see the note on GPUMatrix's is_custom path.
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
# calc_norm_sq / calc_slice_norms duck-typed dispatch
# ---------------------------------------------------------------------------


def test_calc_norm_sq_dispatches_to_duck_typed_method(dense):
    fake = _FakeCenteredMatrix(dense)
    assert calc_norm_sq(fake) == pytest.approx(np.sum(dense**2))


def test_calc_norm_sq_rejects_external_means_alongside_duck_typed_method(dense):
    fake = _FakeCenteredMatrix(dense)
    with pytest.raises(ValueError, match="its own centering"):
        calc_norm_sq(fake, means=np.ones(dense.shape[1]))


def test_calc_norm_sq_allows_all_zero_means_alongside_duck_typed_method(dense):
    fake = _FakeCenteredMatrix(dense)
    assert calc_norm_sq(fake, means=np.zeros(dense.shape[1])) == pytest.approx(
        np.sum(dense**2)
    )


def test_calc_slice_norms_dispatches_to_duck_typed_method(dense):
    fake = _FakeCenteredMatrix(dense)
    idxs = np.array([0, 0, 1, 1, 1, 2])
    n_cond = 3
    expected = np.array([np.linalg.norm(dense[idxs == i]) for i in range(n_cond)])
    np.testing.assert_allclose(calc_slice_norms(fake, None, idxs, n_cond), expected)


def test_calc_slice_norms_rejects_external_means_alongside_duck_typed_method(dense):
    fake = _FakeCenteredMatrix(dense)
    idxs = np.array([0, 0, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="its own centering"):
        calc_slice_norms(fake, np.ones(dense.shape[1]), idxs, 2)


# ---------------------------------------------------------------------------
# GPUMatrix / to_gpu duck-typed dispatch
# ---------------------------------------------------------------------------


def test_gpu_matrix_passes_through_a_custom_type_on_cpu(dense):
    fake = _FakeCenteredMatrix(dense)
    gpu_mat = GPUMatrix(fake, backend="cpu")
    assert gpu_mat.device_mat is fake

    rhs = np.ones((dense.shape[1], 3))
    np.testing.assert_allclose(gpu_mat @ rhs, dense @ rhs)

    lhs = np.ones((3, dense.shape[0]))
    np.testing.assert_allclose(lhs @ gpu_mat, lhs @ dense)


def test_gpu_matrix_uses_to_device_for_a_gpu_backend(dense):
    fake = _FakeDeviceCapableMatrix(dense)
    gpu_mat = GPUMatrix(fake, backend="cupy")

    assert fake.to_device_calls == ["cupy"]
    assert gpu_mat.device_mat is not fake  # the object to_device() returned

    rhs = np.ones((dense.shape[1], 2))
    np.testing.assert_allclose(gpu_mat @ rhs, dense @ rhs)


def test_gpu_matrix_raises_a_clear_error_without_to_device(dense):
    fake = _FakeCenteredMatrix(dense)
    with pytest.raises(TypeError, match="to_device"):
        GPUMatrix(fake, backend="cupy")
