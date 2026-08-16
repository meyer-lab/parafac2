from typing import Any, cast

import numpy as np
from scipy.sparse import csr_array, issparse

_MLX_CSR_SPMM_KERNEL = None
_MLX_CSR_ATOMIC_RSPMM_KERNEL = None

_MLX_METAL_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""


def get_backend(backend: str | None = None) -> str:
    """Return requested or first available backend ('mlx', 'cupy', or 'cpu')."""
    if backend is not None:
        backend_lower = backend.lower()
        if backend_lower in ("mlx", "cupy", "cpu"):
            return backend_lower
        raise ValueError(
            f"Unknown backend '{backend}'. Supported backends: 'mlx', 'cupy', 'cpu'."
        )

    try:
        import mlx.core  # noqa: F401

        return "mlx"
    except ImportError:
        pass

    try:
        import cupy  # noqa: F401

        return "cupy"
    except ImportError:
        pass

    return "cpu"


def _make_mlx_kernel(
    name: str,
    input_names: list[str],
    output_names: list[str],
    source: str,
    atomic_outputs: bool = False,
) -> Any:
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name=name,
        input_names=input_names,
        output_names=output_names,
        source=source,
        atomic_outputs=atomic_outputs,
        header=_MLX_METAL_HEADER,
    )


def _get_mlx_csr_spmm_kernel() -> Any:
    global _MLX_CSR_SPMM_KERNEL
    if _MLX_CSR_SPMM_KERNEL is None:
        source = """
            uint row = thread_position_in_grid.x;
            uint k = thread_position_in_grid.y;
            if (row >= M || k >= K) return;

            Idx start = indptr[row];
            Idx end = indptr[row + 1];
            T acc = 0;
            for (Idx p = start; p < end; p++) {
                Idx col = indices[p];
                acc += data[p] * rhs[col * K + k];
            }
            out[row * K + k] = acc;
        """
        _MLX_CSR_SPMM_KERNEL = _make_mlx_kernel(
            name="csr_spmm",
            input_names=["data", "indices", "indptr", "rhs"],
            output_names=["out"],
            source=source,
        )
    return _MLX_CSR_SPMM_KERNEL


def _get_mlx_csr_atomic_rspmm_kernel() -> Any:
    global _MLX_CSR_ATOMIC_RSPMM_KERNEL
    if _MLX_CSR_ATOMIC_RSPMM_KERNEL is None:
        source = """
            uint k = thread_position_in_grid.x;
            uint row = thread_position_in_grid.y;
            if (k >= K || row >= M) return;

            T val_lhs = lhs[k * M + row];
            if (val_lhs == 0.0f) return;

            Idx start = indptr[row];
            Idx end = indptr[row + 1];
            for (Idx p = start; p < end; p++) {
                Idx col = indices[p];
                T prod = val_lhs * data[p];
                atomic_fetch_add_explicit(&out[k * N + col], prod, memory_order_relaxed);
            }
        """
        _MLX_CSR_ATOMIC_RSPMM_KERNEL = _make_mlx_kernel(
            name="dense_csr_spmm_atomic",
            input_names=["lhs", "data", "indices", "indptr"],
            output_names=["out"],
            source=source,
            atomic_outputs=True,
        )
    return _MLX_CSR_ATOMIC_RSPMM_KERNEL


def _csr_to_mlx(mat_csr: csr_array) -> tuple[Any, Any, Any]:
    import mlx.core as mx

    mx_data = mx.array(mat_csr.data.astype(np.float32, copy=False))
    mx_indices = mx.array(mat_csr.indices.astype(np.int32, copy=False))
    mx_indptr = mx.array(mat_csr.indptr.astype(np.int32, copy=False))
    return mx_data, mx_indices, mx_indptr


def _mlx_to_numpy(mx_out: Any, orig_dtype: np.dtype, is_1d: bool) -> np.ndarray:
    import mlx.core as mx

    mx.eval(mx_out)
    res_arr = np.asarray(mx_out).astype(orig_dtype, copy=False)
    return res_arr.ravel() if is_1d else res_arr


def _matmul_mlx(mat: np.ndarray | csr_array, rhs: np.ndarray) -> np.ndarray:
    import mlx.core as mx

    orig_dtype = getattr(rhs, "dtype", mat.dtype)
    M, _N = mat.shape

    if issparse(mat):
        mat_csr = cast("csr_array", mat)
        rhs_2d = rhs[:, None] if rhs.ndim == 1 else rhs
        K = rhs_2d.shape[1]

        mx_data, mx_indices, mx_indptr = _csr_to_mlx(mat_csr)
        mx_rhs = mx.array(rhs_2d.astype(np.float32, copy=False))

        kernel = _get_mlx_csr_spmm_kernel()
        tg_m = min(M, 16)
        tg_k = min(K, 16)
        out = kernel(
            inputs=[mx_data, mx_indices, mx_indptr, mx_rhs],
            template=[("T", mx.float32), ("Idx", mx.int32), ("M", M), ("K", K)],
            grid=(M, K, 1),
            threadgroup=(tg_m, tg_k, 1),
            output_shapes=[(M, K)],
            output_dtypes=[mx.float32],
        )
        return _mlx_to_numpy(out[0], orig_dtype, rhs.ndim == 1)
    else:
        mx_mat = mx.array(mat)
        mx_rhs = mx.array(rhs)
        mx_res = mx_mat @ mx_rhs
        return _mlx_to_numpy(mx_res, orig_dtype, rhs.ndim == 1)


def _rmatmul_mlx(lhs: np.ndarray, mat: np.ndarray | csr_array) -> np.ndarray:
    import mlx.core as mx

    orig_dtype = getattr(lhs, "dtype", mat.dtype)
    M, N = mat.shape

    if issparse(mat):
        mat_csr = cast("csr_array", mat)
        lhs_2d = lhs[None, :] if lhs.ndim == 1 else lhs
        K = lhs_2d.shape[0]

        mx_data, mx_indices, mx_indptr = _csr_to_mlx(mat_csr)
        mx_lhs = mx.array(lhs_2d.astype(np.float32, copy=False))

        kernel = _get_mlx_csr_atomic_rspmm_kernel()
        tg_k = min(K, 16)
        tg_m = min(M, 16)
        out = kernel(
            inputs=[mx_lhs, mx_data, mx_indices, mx_indptr],
            template=[
                ("T", mx.float32),
                ("Idx", mx.int32),
                ("K", K),
                ("M", M),
                ("N", N),
            ],
            grid=(K, M, 1),
            threadgroup=(tg_k, tg_m, 1),
            output_shapes=[(K, N)],
            output_dtypes=[mx.float32],
            init_value=0.0,
        )
        return _mlx_to_numpy(out[0], orig_dtype, lhs.ndim == 1)
    else:
        mx_lhs = mx.array(lhs)
        mx_mat = mx.array(mat)
        mx_res = mx_lhs @ mx_mat
        return _mlx_to_numpy(mx_res, orig_dtype, lhs.ndim == 1)


def _to_cupy_matrix(mat: np.ndarray | csr_array) -> Any:
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse

    if issparse(mat):
        mat_csr = cast("csr_array", mat)
        cp_data = cp.asarray(mat_csr.data)
        cp_indices = cp.asarray(mat_csr.indices)
        cp_indptr = cp.asarray(mat_csr.indptr)
        return cpsparse.csr_matrix(
            (cp_data, cp_indices, cp_indptr), shape=mat_csr.shape
        )
    return cp.asarray(mat)


def _matmul_cupy(mat: np.ndarray | csr_array, rhs: np.ndarray) -> np.ndarray:
    import cupy as cp

    cp_mat = _to_cupy_matrix(mat)
    cp_rhs = cp.asarray(rhs)
    cp_res = cp_mat @ cp_rhs
    return cp.asnumpy(cp_res)


def _rmatmul_cupy(lhs: np.ndarray, mat: np.ndarray | csr_array) -> np.ndarray:
    import cupy as cp

    cp_lhs = cp.asarray(lhs)
    cp_mat = _to_cupy_matrix(mat)
    cp_res = cp_lhs @ cp_mat
    return cp.asnumpy(cp_res)


class SampleArray:
    """
    Wrapper for a single sample matrix (csr_array or np.ndarray) and its gene means.
    Automatically performs mean-centering during left and right matrix multiplications.
    """

    __array_priority__ = 1000

    def __init__(self, mat: np.ndarray | csr_array, means: np.ndarray) -> None:
        if issparse(mat):
            self.mat = csr_array(mat)
        else:
            self.mat = np.asarray(mat)
        self.means = np.asarray(means).ravel()

    @property
    def shape(self) -> tuple[int, int]:
        return self.mat.shape

    @property
    def dtype(self) -> np.dtype:
        return self.mat.dtype

    @property
    def ndim(self) -> int:
        return 2

    def __len__(self) -> int:
        return self.mat.shape[0]

    def toarray(self) -> np.ndarray:
        """Return the dense, mean-centered matrix."""
        dense = (
            cast("csr_array", self.mat).toarray() if issparse(self.mat) else self.mat
        )
        return dense - self.means

    def norm_sq(self) -> float:
        """Return the squared Frobenius norm of the mean-centered matrix."""
        if issparse(self.mat):
            mat_csr = cast("csr_array", self.mat)
            M = mat_csr.shape[0]
            term1 = np.sum(mat_csr.data**2)
            term2 = -2.0 * np.sum(mat_csr.data * self.means[mat_csr.indices])
            term3 = M * np.sum(self.means**2)
            return float(term1 + term2 + term3)
        return float(np.sum((self.mat - self.means) ** 2))

    def __matmul__(self, rhs: np.ndarray, backend: str | None = None) -> np.ndarray:
        """
        Left matrix multiplication: self @ rhs
        Computes (self.mat - means) @ rhs = self.mat @ rhs - means @ rhs
        """
        chosen_backend = get_backend(backend)
        if chosen_backend == "mlx":
            res_arr = _matmul_mlx(self.mat, rhs)
        elif chosen_backend == "cupy":
            res_arr = _matmul_cupy(self.mat, rhs)
        else:
            res_arr = self.mat @ rhs

        res_arr -= self.means @ rhs
        return res_arr

    def __rmatmul__(self, lhs: np.ndarray, backend: str | None = None) -> np.ndarray:
        """
        Right matrix multiplication: lhs @ self
        Computes lhs @ (self.mat - means) = lhs @ self.mat -
        outer(sum(lhs, axis=1), means)
        """
        chosen_backend = get_backend(backend)
        if chosen_backend == "mlx":
            res_arr = _rmatmul_mlx(lhs, self.mat)
        elif chosen_backend == "cupy":
            res_arr = _rmatmul_cupy(lhs, self.mat)
        else:
            res_arr = lhs @ self.mat

        if lhs.ndim == 2:
            row_sums = np.sum(lhs, axis=1)
            res_arr -= np.outer(row_sums, self.means)
        else:
            res_arr -= np.sum(lhs) * self.means
        return res_arr
