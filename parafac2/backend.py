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
        import cupy  # noqa: F401

        return "cupy"
    except ImportError:
        pass

    try:
        import mlx.core  # noqa: F401

        return "mlx"
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


def _to_mlx_matrix(mat: np.ndarray | csr_array) -> Any:
    import mlx.core as mx

    if issparse(mat):
        return _csr_to_mlx(cast("csr_array", mat))
    return mx.array(mat)


def _matmul_mlx(
    device_mat: Any, rhs: np.ndarray, is_sparse: bool, shape: tuple[int, int]
) -> np.ndarray:
    import mlx.core as mx

    orig_dtype = rhs.dtype
    M, _N = shape

    if is_sparse:
        mx_data, mx_indices, mx_indptr = device_mat
        rhs_2d = rhs[:, None] if rhs.ndim == 1 else rhs
        K = rhs_2d.shape[1]

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
        mx_rhs = mx.array(rhs)
        mx_res = device_mat @ mx_rhs
        return _mlx_to_numpy(mx_res, orig_dtype, rhs.ndim == 1)


def _rmatmul_mlx(
    lhs: np.ndarray, device_mat: Any, is_sparse: bool, shape: tuple[int, int]
) -> np.ndarray:
    import mlx.core as mx

    orig_dtype = lhs.dtype
    M, N = shape

    if is_sparse:
        mx_data, mx_indices, mx_indptr = device_mat
        lhs_2d = lhs[None, :] if lhs.ndim == 1 else lhs
        K = lhs_2d.shape[0]

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
        mx_res = mx_lhs @ device_mat
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


def _matmul_cupy(cp_mat: Any, rhs: np.ndarray) -> np.ndarray:
    import cupy as cp

    cp_rhs = cp.asarray(rhs)
    cp_res = cp_mat @ cp_rhs
    return cp.asnumpy(cp_res)


def _rmatmul_cupy(lhs: np.ndarray, cp_mat: Any) -> np.ndarray:
    import cupy as cp

    cp_lhs = cp.asarray(lhs)
    cp_res = cp_lhs @ cp_mat
    return cp.asnumpy(cp_res)


class GPUMatrix:
    """
    Wrapper for a single matrix (csr_array or np.ndarray) stored on GPU memory
    (CuPy or MLX) or CPU. Evaluates matrix products on the device and returns
    results as NumPy ndarrays.
    """

    __array_priority__ = 1000

    def __init__(self, mat: np.ndarray | csr_array, backend: str | None = None) -> None:
        self.backend = get_backend(backend)
        self.shape = mat.shape
        self.dtype = mat.dtype
        self.is_sparse = issparse(mat)

        if self.backend == "cupy":
            self.device_mat = _to_cupy_matrix(mat)
        elif self.backend == "mlx":
            self.device_mat = _to_mlx_matrix(mat)
        else:
            self.device_mat = mat

    def matmul(self, rhs: np.ndarray) -> np.ndarray:
        """Compute self @ rhs and return NumPy array."""
        if self.backend == "cupy":
            return _matmul_cupy(self.device_mat, rhs)
        elif self.backend == "mlx":
            return _matmul_mlx(
                self.device_mat, rhs, is_sparse=self.is_sparse, shape=self.shape
            )
        return self.device_mat @ rhs

    def rmatmul(self, lhs: np.ndarray) -> np.ndarray:
        """Compute lhs @ self and return NumPy array."""
        if self.backend == "cupy":
            return _rmatmul_cupy(lhs, self.device_mat)
        elif self.backend == "mlx":
            return _rmatmul_mlx(
                lhs, self.device_mat, is_sparse=self.is_sparse, shape=self.shape
            )
        return lhs @ self.device_mat

    def __matmul__(self, rhs: np.ndarray) -> np.ndarray:
        return self.matmul(rhs)

    def __rmatmul__(self, lhs: np.ndarray) -> np.ndarray:
        return self.rmatmul(lhs)


def to_gpu(
    mat: np.ndarray | csr_array, backend: str | None = None
) -> GPUMatrix | np.ndarray | csr_array:
    """
    Transfer matrix to GPU memory if CuPy or MLX is requested/available,
    returning a GPUMatrix wrapper. Otherwise returns the CPU matrix as-is.
    """
    chosen = get_backend(backend)
    if chosen == "cpu":
        return mat
    return GPUMatrix(mat, backend=chosen)
