"""
GPU/CPU backend abstraction for matrix operations.

Provides a unified interface for performing dense and sparse (CSR) matrix
multiplications on CPU (NumPy), Apple GPUs (MLX), or NVIDIA GPUs (CuPy). This
lets the PARAFAC2 fit run its matrix products on whichever accelerator is
available without copying data through an intermediate common format.
"""

from typing import Any, cast

import numpy as np
from scipy.sparse import csr_array, issparse

_MLX_CSR_SPMM_KERNEL = None
_MLX_CSR_ATOMIC_RSPMM_KERNEL = None
_MKL_DOT: Any = False

_MLX_METAL_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""


def get_backend(backend: str | None = None) -> str:
    """Return the requested backend, or auto-detect the first available one.

    Parameters
    ----------
    backend : str, optional
        One of ``'mlx'``, ``'cupy'``, or ``'cpu'``. If ``None``, the first
        available accelerator is chosen by attempting to import ``cupy``
        then ``mlx.core``, falling back to ``'cpu'`` if neither is
        installed.

    Returns
    -------
    str
        The resolved backend name: ``'mlx'``, ``'cupy'``, or ``'cpu'``.

    Raises
    ------
    ValueError
        If ``backend`` is given but is not one of the supported names.
    """
    if backend is not None:
        backend_lower = backend.lower()
        if backend_lower in ("mlx", "cupy", "cpu"):
            return backend_lower
        raise ValueError(
            f"Unknown backend '{backend}'. Supported backends: 'mlx', 'cupy', 'cpu'."
        )

    try:
        import cupy  # noqa: F401  # ty: ignore[unresolved-import]

        return "cupy"
    except ImportError:
        pass

    try:
        import mlx.core  # noqa: F401  # ty: ignore[unresolved-import]

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
    """Compile an MLX Metal kernel from source.

    Parameters
    ----------
    name : str
        Name to register the kernel under.
    input_names : list[str]
        Names of the kernel's input buffers, matching the Metal source.
    output_names : list[str]
        Names of the kernel's output buffers.
    source : str
        Metal shader source implementing the kernel body.
    atomic_outputs : bool, default False
        Whether the output buffers must be written using atomic operations
        (needed when multiple threads accumulate into the same output cell).

    Returns
    -------
    Any
        The compiled ``mx.fast.metal_kernel`` callable.
    """
    import mlx.core as mx  # ty: ignore[unresolved-import]

    return mx.fast.metal_kernel(
        name=name,
        input_names=input_names,
        output_names=output_names,
        source=source,
        atomic_outputs=atomic_outputs,
        header=_MLX_METAL_HEADER,
    )


def _get_mlx_csr_spmm_kernel() -> Any:
    """Return the cached MLX kernel for sparse (CSR) @ dense multiplication.

    Compiles the kernel on first use and caches it in the module-level
    ``_MLX_CSR_SPMM_KERNEL`` global for subsequent calls.

    Returns
    -------
    Any
        The compiled ``csr_spmm`` MLX kernel.
    """
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
    """Return the cached MLX kernel for dense @ sparse (CSR) multiplication.

    The kernel accumulates into the (dense) output using atomic adds, since
    multiple threads may write to the same output column. Compiles the
    kernel on first use and caches it in the module-level
    ``_MLX_CSR_ATOMIC_RSPMM_KERNEL`` global for subsequent calls.

    Returns
    -------
    Any
        The compiled ``dense_csr_spmm_atomic`` MLX kernel.
    """
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
    """Move a SciPy CSR array's components onto the MLX device.

    Parameters
    ----------
    mat_csr : csr_array
        The sparse matrix to transfer.

    Returns
    -------
    tuple[Any, Any, Any]
        The ``(data, indices, indptr)`` MLX arrays backing the CSR matrix.
    """
    import mlx.core as mx  # ty: ignore[unresolved-import]

    mx_data = mx.array(mat_csr.data.astype(np.float32, copy=False))
    mx_indices = mx.array(mat_csr.indices.astype(np.int32, copy=False))
    mx_indptr = mx.array(mat_csr.indptr.astype(np.int32, copy=False))
    return mx_data, mx_indices, mx_indptr


def _mlx_to_numpy(mx_out: Any, orig_dtype: np.dtype, is_1d: bool) -> np.ndarray:
    """Evaluate an MLX array and convert it to a NumPy array.

    Parameters
    ----------
    mx_out : Any
        The (possibly lazy) MLX array to materialize.
    orig_dtype : np.dtype
        The dtype the result should be cast back to.
    is_1d : bool
        Whether the result should be raveled to a 1-D array (used when the
        original operand was a 1-D vector promoted to 2-D for the kernel).

    Returns
    -------
    np.ndarray
        The result as a NumPy array of ``orig_dtype``.
    """
    import mlx.core as mx  # ty: ignore[unresolved-import]

    mx.eval(mx_out)
    res_arr = np.asarray(mx_out).astype(orig_dtype, copy=False)
    return res_arr.ravel() if is_1d else res_arr


def _to_mlx_matrix(mat: np.ndarray | csr_array) -> Any:
    """Move a dense or CSR matrix onto the MLX device.

    Parameters
    ----------
    mat : np.ndarray | csr_array
        The matrix to transfer.

    Returns
    -------
    Any
        An ``mx.array`` for dense input, or the ``(data, indices, indptr)``
        MLX array tuple for sparse input.
    """
    import mlx.core as mx  # ty: ignore[unresolved-import]

    if issparse(mat):
        return _csr_to_mlx(cast("csr_array", mat))
    return mx.array(mat)


def _matmul_mlx(
    device_mat: Any, rhs: np.ndarray, is_sparse: bool, shape: tuple[int, int]
) -> np.ndarray:
    """Compute ``device_mat @ rhs`` on the MLX backend.

    Dispatches to the CSR sparse-matrix-multiply kernel when ``is_sparse``
    is set, otherwise uses MLX's native dense matmul.

    Parameters
    ----------
    device_mat : Any
        The left-hand matrix already resident on the MLX device, as
        returned by :func:`_to_mlx_matrix`.
    rhs : np.ndarray
        The right-hand operand (1-D or 2-D NumPy array).
    is_sparse : bool
        Whether ``device_mat`` represents a sparse CSR matrix.
    shape : tuple[int, int]
        The logical ``(rows, cols)`` shape of ``device_mat``.

    Returns
    -------
    np.ndarray
        The product, as a NumPy array with the same dtype as ``rhs``.
    """
    import mlx.core as mx  # ty: ignore[unresolved-import]

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
    """Compute ``lhs @ device_mat`` on the MLX backend.

    Dispatches to the atomic CSR sparse-matrix-multiply kernel when
    ``is_sparse`` is set, otherwise uses MLX's native dense matmul.

    Parameters
    ----------
    lhs : np.ndarray
        The left-hand operand (1-D or 2-D NumPy array).
    device_mat : Any
        The right-hand matrix already resident on the MLX device, as
        returned by :func:`_to_mlx_matrix`.
    is_sparse : bool
        Whether ``device_mat`` represents a sparse CSR matrix.
    shape : tuple[int, int]
        The logical ``(rows, cols)`` shape of ``device_mat``.

    Returns
    -------
    np.ndarray
        The product, as a NumPy array with the same dtype as ``lhs``.
    """
    import mlx.core as mx  # ty: ignore[unresolved-import]

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
    """Move a dense or CSR matrix onto the CuPy device.

    Parameters
    ----------
    mat : np.ndarray | csr_array
        The matrix to transfer.

    Returns
    -------
    Any
        A CuPy ``ndarray`` for dense input, or a
        ``cupyx.scipy.sparse.csr_matrix`` for sparse input.
    """
    import cupy as cp  # ty: ignore[unresolved-import]
    import cupyx.scipy.sparse as cpsparse  # ty: ignore[unresolved-import]

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
    """Compute ``cp_mat @ rhs`` on the CuPy backend.

    Parameters
    ----------
    cp_mat : Any
        The left-hand matrix already resident on the CuPy device, as
        returned by :func:`_to_cupy_matrix`.
    rhs : np.ndarray
        The right-hand operand.

    Returns
    -------
    np.ndarray
        The product, transferred back to host memory as a NumPy array.
    """
    import cupy as cp  # ty: ignore[unresolved-import]

    cp_rhs = cp.asarray(rhs)
    cp_res = cp_mat @ cp_rhs
    return cp.asnumpy(cp_res)


def _rmatmul_cupy(lhs: np.ndarray, cp_mat: Any) -> np.ndarray:
    """Compute ``lhs @ cp_mat`` on the CuPy backend.

    Parameters
    ----------
    lhs : np.ndarray
        The left-hand operand.
    cp_mat : Any
        The right-hand matrix already resident on the CuPy device, as
        returned by :func:`_to_cupy_matrix`.

    Returns
    -------
    np.ndarray
        The product, transferred back to host memory as a NumPy array.
    """
    import cupy as cp  # ty: ignore[unresolved-import]

    cp_lhs = cp.asarray(lhs)
    cp_res = cp_lhs @ cp_mat
    return cp.asnumpy(cp_res)


class GPUMatrix:
    """
    Wrapper for a single matrix (csr_array or np.ndarray) stored on GPU memory
    (CuPy or MLX) or CPU. Evaluates matrix products on the device and returns
    results as NumPy ndarrays.

    Parameters
    ----------
    mat : np.ndarray | csr_array
        The matrix to wrap and transfer to the selected device.
    backend : str, optional
        One of ``'mlx'``, ``'cupy'``, or ``'cpu'``. If ``None``, the first
        available accelerator is auto-detected (see :func:`get_backend`).
    """

    __array_priority__ = 1000

    def __init__(self, mat: np.ndarray | csr_array, backend: str | None = None) -> None:
        """Transfer ``mat`` to the resolved backend's device memory."""
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
        """Compute ``self @ rhs`` on the wrapped device.

        Parameters
        ----------
        rhs : np.ndarray
            The right-hand operand.

        Returns
        -------
        np.ndarray
            The product, as a NumPy array.
        """
        if self.backend == "cupy":
            return _matmul_cupy(self.device_mat, rhs)
        elif self.backend == "mlx":
            return _matmul_mlx(
                self.device_mat, rhs, is_sparse=self.is_sparse, shape=self.shape
            )
        return self.device_mat @ rhs

    def rmatmul(self, lhs: np.ndarray) -> np.ndarray:
        """Compute ``lhs @ self`` on the wrapped device.

        Parameters
        ----------
        lhs : np.ndarray
            The left-hand operand.

        Returns
        -------
        np.ndarray
            The product, as a NumPy array.
        """
        if self.backend == "cupy":
            return _rmatmul_cupy(lhs, self.device_mat)
        elif self.backend == "mlx":
            return _rmatmul_mlx(
                lhs, self.device_mat, is_sparse=self.is_sparse, shape=self.shape
            )
        return lhs @ self.device_mat

    def __matmul__(self, rhs: np.ndarray) -> np.ndarray:
        """Operator form of :meth:`matmul`, enabling ``gpu_matrix @ rhs``."""
        return self.matmul(rhs)

    def __rmatmul__(self, lhs: np.ndarray) -> np.ndarray:
        """Operator form of :meth:`rmatmul`, enabling ``lhs @ gpu_matrix``."""
        return self.rmatmul(lhs)


def _get_mkl_dot() -> Any:
    """Return ``sparse_dot_mkl.dot_product_mkl`` if importable, else ``None``.

    SciPy's sparse-times-dense kernels are single-threaded, which dominates
    the PARAFAC2 fit on large datasets. When the optional ``sparse-dot-mkl``
    package is installed (``pip install 'parafac2[mkl]'``) its multithreaded
    MKL kernels are used instead. The lookup is cached in the module-level
    ``_MKL_DOT`` global.
    """
    global _MKL_DOT
    if _MKL_DOT is False:
        try:
            from sparse_dot_mkl import (  # ty: ignore[unresolved-import]
                dot_product_mkl,
            )

            _MKL_DOT = dot_product_mkl
        except ImportError:
            _MKL_DOT = None
    return _MKL_DOT


def _mkl_compatible(mat: Any, dense: np.ndarray) -> bool:
    """Whether ``mat``/``dense`` can be handed to MKL's sparse kernels.

    MKL requires a CPU-resident CSR/CSC matrix and a dense operand that
    shares its (single- or double-precision) dtype and is contiguous.
    """
    if not issparse(mat) or _get_mkl_dot() is None:
        return False
    return (
        mat.dtype == dense.dtype
        and mat.dtype in (np.float32, np.float64)
        and dense.flags.c_contiguous
    )


def matmul(mat: Any, rhs: np.ndarray) -> np.ndarray:
    """Compute ``mat @ rhs``, dispatching to the fastest available kernel.

    Parameters
    ----------
    mat : Any
        A :class:`GPUMatrix`, SciPy sparse matrix, or dense NumPy array.
    rhs : np.ndarray
        The dense right-hand operand.

    Returns
    -------
    np.ndarray
        The product ``mat @ rhs``.

    Notes
    -----
    ``rhs`` should already share ``mat``'s dtype. Handing a float64 ``rhs``
    to a float32 sparse ``mat`` makes SciPy upcast the *whole* sparse matrix,
    which for single-cell-sized data is both slow and memory-hostile; see
    :func:`~parafac2.utils.calc_W`.
    """
    if isinstance(mat, GPUMatrix):
        return mat.matmul(rhs)
    if _mkl_compatible(mat, rhs):
        return _get_mkl_dot()(mat, rhs)
    return mat @ rhs


def rmatmul(lhs: np.ndarray, mat: Any) -> np.ndarray:
    """Compute ``lhs @ mat``, dispatching to the fastest available kernel.

    Parameters
    ----------
    lhs : np.ndarray
        The dense left-hand operand.
    mat : Any
        A :class:`GPUMatrix`, SciPy sparse matrix, or dense NumPy array.

    Returns
    -------
    np.ndarray
        The product ``lhs @ mat``.
    """
    if isinstance(mat, GPUMatrix):
        return mat.rmatmul(lhs)
    if _mkl_compatible(mat, lhs):
        return _get_mkl_dot()(lhs, mat)
    return lhs @ mat


def matrix_dtype(mat: Any) -> np.dtype:
    """Return the dtype of a :class:`GPUMatrix`, sparse matrix, or ndarray."""
    return np.dtype(getattr(mat, "dtype", np.float64))


def to_gpu(
    mat: np.ndarray | csr_array, backend: str | None = None
) -> GPUMatrix | np.ndarray | csr_array:
    """
    Transfer matrix to GPU memory if CuPy or MLX is requested/available,
    returning a GPUMatrix wrapper. Otherwise returns the CPU matrix as-is.

    Parameters
    ----------
    mat : np.ndarray | csr_array
        The matrix to (optionally) transfer.
    backend : str, optional
        One of ``'mlx'``, ``'cupy'``, or ``'cpu'``. If ``None``, the first
        available accelerator is auto-detected (see :func:`get_backend`).

    Returns
    -------
    GPUMatrix | np.ndarray | csr_array
        A :class:`GPUMatrix` wrapping ``mat`` if a GPU backend was resolved,
        otherwise ``mat`` unchanged.
    """
    chosen = get_backend(backend)
    if chosen == "cpu":
        return mat
    return GPUMatrix(mat, backend=chosen)
