"""Property-based tests for the ``backend`` compute-dispatch module.

Covers backend-name resolution across arbitrary casing/whitespace, the
device-memory-accounting formulas, and (when the corresponding accelerator
is actually available) that GPU-backed matmul/rmatmul agree with plain NumPy
on arbitrary shapes, sparsity patterns, and dtypes.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.sparse import csr_array

from ..backend import _VALID_BACKENDS, csr_device_bytes, get_backend
from ..matrix import GPUMatrix
from .strategies import dense_matrix_with_sparsity, real_arrays, to_csr

# ---------------------------------------------------------------------------
# get_backend
# ---------------------------------------------------------------------------


def _wrap_case_and_whitespace(name: str, upper: bool, pad: str) -> str:
    return pad + (name.upper() if upper else name) + pad


@given(
    name=st.sampled_from(_VALID_BACKENDS),
    upper=st.booleans(),
    pad=st.sampled_from(["", " ", "  ", "\t"]),
)
@settings(max_examples=50)
def test_get_backend_accepts_any_casing_or_whitespace_of_a_valid_name(name, upper, pad):
    mangled = _wrap_case_and_whitespace(name, upper, pad)
    assert get_backend(mangled) == name


@given(text=st.text(min_size=1, max_size=20))
@settings(max_examples=200)
def test_get_backend_rejects_anything_that_is_not_a_known_backend(text):
    if text.strip().lower() in _VALID_BACKENDS:
        return  # not the property under test; covered by the test above
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend(text)


# ---------------------------------------------------------------------------
# csr_device_bytes
# ---------------------------------------------------------------------------


@given(mat=dense_matrix_with_sparsity(min_rows=1, max_rows=30, min_cols=1, max_cols=30))
@settings(max_examples=50)
def test_device_bytes_sparse_matches_int32_index_formula(mat):
    """Below the int32 boundary, both `indices` and `indptr` cost 4 bytes/entry."""
    mat_csr = to_csr(mat)
    nnz = mat_csr.data.size
    expected = mat_csr.data.nbytes + (nnz + mat_csr.shape[0] + 1) * 4
    assert csr_device_bytes(mat_csr) == expected


# ---------------------------------------------------------------------------
# to_gpu matmul/rmatmul: real-accelerator agreement with plain NumPy
# ---------------------------------------------------------------------------


def _backend_available(name: str) -> bool:
    if name == "cpu":
        return True
    if name == "mlx":
        try:
            import mlx.core  # noqa: F401  # ty: ignore[unresolved-import]

            return True
        except ImportError:
            return False
    if name == "cupy":
        try:
            import cupy  # ty: ignore[unresolved-import]

            return cupy.cuda.runtime.getDeviceCount() > 0
        except Exception:  # noqa: BLE001
            return False
    return False


@pytest.mark.parametrize("backend", ["cpu", "mlx", "cupy"])
@given(
    mat=dense_matrix_with_sparsity(min_rows=1, max_rows=20, min_cols=1, max_cols=15),
    sparse=st.booleans(),
    rhs_cols=st.integers(1, 6),
    data=st.data(),
)
@settings(max_examples=25, deadline=None)
def test_accelerated_matmul_agrees_with_numpy(backend, mat, sparse, rhs_cols, data):
    """matmul/rmatmul on a real accelerator device must reproduce plain NumPy,
    for both dense and sparse operands. Skips if that accelerator is not
    actually usable in this environment (the CI matrix runs `mlx` on macOS
    and `cupy` on the self-hosted CUDA runner, so each still gets covered)."""
    if not _backend_available(backend):
        pytest.skip(f"backend '{backend}' is not usable in this environment")

    mat32 = mat.astype(np.float32)
    lhs_mat = csr_array(mat32) if sparse else mat32

    gpu_mat = GPUMatrix(lhs_mat, backend=backend)

    rhs = data.draw(
        real_arrays((mat32.shape[1], rhs_cols), dtype=np.float32), label="rhs"
    )
    np.testing.assert_allclose(gpu_mat @ rhs, mat32 @ rhs, rtol=1e-4, atol=1e-4)

    lhs = data.draw(
        real_arrays((rhs_cols, mat32.shape[0]), dtype=np.float32), label="lhs"
    )
    np.testing.assert_allclose(lhs @ gpu_mat, lhs @ mat32, rtol=1e-4, atol=1e-4)
