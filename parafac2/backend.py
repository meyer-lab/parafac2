"""
Low-level device primitives for CPU, Apple GPU (MLX), and NVIDIA GPU (CuPy).

This module knows about *devices*, not about matrices: it resolves which
backend to run on, accounts for device memory, and provides the transfer and
kernel routines for dense and CSR operands on each backend. Nothing here
inspects the type of the user's data.

The matrix types that consume these primitives -- and the duck-typed contract
the rest of the package talks to -- live in :mod:`parafac2.matrix`.
"""

import os
from typing import Any

import numpy as np

# Environment variable that forces a backend, overriding auto-detection.
BACKEND_ENV_VAR = "PARAFAC2_BACKEND"

_VALID_BACKENDS = ("mlx", "cupy", "cpu")

# Fraction of free device memory a single transfer may claim before the
# CuPy backend switches to managed memory.
DEVICE_MEMORY_HEADROOM = 0.8

# Set once the managed-memory allocator has been installed.
_managed_allocator_installed = False

_MLX_CSR_SPMM_KERNEL = None
_MLX_CSR_ATOMIC_RSPMM_KERNEL = None
_MKL_DOT: Any = False

_MLX_METAL_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def _cuda_is_usable() -> bool:
    """Whether CuPy is installed and a CUDA device is actually present."""
    try:
        import cupy  # ty: ignore[unresolved-import]

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


def get_backend(backend: str | None = None) -> str:
    """Return the requested backend, or auto-detect the first available one.

    Parameters
    ----------
    backend : str, optional
        One of ``'mlx'``, ``'cupy'``, or ``'cpu'``. ``None`` consults
        ``PARAFAC2_BACKEND`` and then auto-detects.

    Returns
    -------
    str
        The resolved backend name: ``'mlx'``, ``'cupy'``, or ``'cpu'``.
    """
    if backend is None:
        backend = os.environ.get(BACKEND_ENV_VAR) or None
        source = f"{BACKEND_ENV_VAR}="
    else:
        source = "backend="

    if backend is not None:
        backend_lower = backend.strip().lower()
        if backend_lower in _VALID_BACKENDS:
            return backend_lower
        raise ValueError(
            f"Unknown backend '{backend}' (from {source}{backend!r}). "
            f"Supported backends: 'mlx', 'cupy', 'cpu'."
        )

    if _cuda_is_usable():
        return "cupy"

    try:
        import mlx.core  # noqa: F401  # ty: ignore[unresolved-import]

        return "mlx"
    except ImportError:
        pass

    return "cpu"


# ---------------------------------------------------------------------------
# MLX (Apple GPU)
# ---------------------------------------------------------------------------


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


def _mlx_to_numpy(mx_out: Any, is_1d: bool) -> np.ndarray:
    """Evaluate an MLX array and convert it to a NumPy array.

    Parameters
    ----------
    mx_out : Any
        The (possibly lazy) MLX array to materialize.
    is_1d : bool
        Whether the result should be raveled to a 1-D array (used when the
        original operand was a 1-D vector promoted to 2-D for the kernel).

    Returns
    -------
    np.ndarray
        The result as a NumPy array.
    """
    import mlx.core as mx  # ty: ignore[unresolved-import]

    mx.eval(mx_out)
    res_arr = np.asarray(mx_out)
    return res_arr.ravel() if is_1d else res_arr


def to_mlx_dense(mat: np.ndarray) -> Any:
    """Move a dense matrix onto the MLX device."""
    import mlx.core as mx  # ty: ignore[unresolved-import]

    return mx.array(mat)


def to_mlx_csr(mat_csr: Any) -> tuple[Any, Any, Any]:
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


def mlx_dense_matmul(
    device_mat: Any, rhs: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Compute ``device_mat @ rhs`` for a dense MLX-resident matrix."""
    import mlx.core as mx  # ty: ignore[unresolved-import]

    return _mlx_to_numpy(device_mat @ mx.array(rhs), rhs.ndim == 1)


def mlx_dense_rmatmul(
    lhs: np.ndarray, device_mat: Any, shape: tuple[int, int]
) -> np.ndarray:
    """Compute ``lhs @ device_mat`` for a dense MLX-resident matrix."""
    import mlx.core as mx  # ty: ignore[unresolved-import]

    return _mlx_to_numpy(mx.array(lhs) @ device_mat, lhs.ndim == 1)


def mlx_csr_matmul(
    device_mat: tuple[Any, Any, Any], rhs: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Compute ``device_mat @ rhs`` via the MLX CSR sparse-matmul kernel.

    Parameters
    ----------
    device_mat : tuple[Any, Any, Any]
        The ``(data, indices, indptr)`` MLX arrays from :func:`to_mlx_csr`.
    rhs : np.ndarray
        The right-hand operand (1-D or 2-D NumPy array).
    shape : tuple[int, int]
        The logical ``(rows, cols)`` shape of the sparse matrix.
    """
    import mlx.core as mx  # ty: ignore[unresolved-import]

    M, _N = shape
    mx_data, mx_indices, mx_indptr = device_mat
    rhs_2d = rhs[:, None] if rhs.ndim == 1 else rhs
    K = rhs_2d.shape[1]

    mx_rhs = mx.array(rhs_2d.astype(np.float32, copy=False))
    out = _get_mlx_csr_spmm_kernel()(
        inputs=[mx_data, mx_indices, mx_indptr, mx_rhs],
        template=[("T", mx.float32), ("Idx", mx.int32), ("M", M), ("K", K)],
        grid=(M, K, 1),
        threadgroup=(min(M, 16), min(K, 16), 1),
        output_shapes=[(M, K)],
        output_dtypes=[mx.float32],
    )
    return _mlx_to_numpy(out[0], rhs.ndim == 1)


def mlx_csr_rmatmul(
    lhs: np.ndarray, device_mat: tuple[Any, Any, Any], shape: tuple[int, int]
) -> np.ndarray:
    """Compute ``lhs @ device_mat`` via the atomic MLX CSR sparse-matmul kernel.

    Parameters
    ----------
    lhs : np.ndarray
        The left-hand operand (1-D or 2-D NumPy array).
    device_mat : tuple[Any, Any, Any]
        The ``(data, indices, indptr)`` MLX arrays from :func:`to_mlx_csr`.
    shape : tuple[int, int]
        The logical ``(rows, cols)`` shape of the sparse matrix.
    """
    import mlx.core as mx  # ty: ignore[unresolved-import]

    M, N = shape
    mx_data, mx_indices, mx_indptr = device_mat
    lhs_2d = lhs[None, :] if lhs.ndim == 1 else lhs
    K = lhs_2d.shape[0]

    mx_lhs = mx.array(lhs_2d.astype(np.float32, copy=False))
    out = _get_mlx_csr_atomic_rspmm_kernel()(
        inputs=[mx_lhs, mx_data, mx_indices, mx_indptr],
        template=[
            ("T", mx.float32),
            ("Idx", mx.int32),
            ("K", K),
            ("M", M),
            ("N", N),
        ],
        grid=(K, M, 1),
        threadgroup=(min(K, 16), min(M, 16), 1),
        output_shapes=[(K, N)],
        output_dtypes=[mx.float32],
        init_value=0.0,
    )
    return _mlx_to_numpy(out[0], lhs.ndim == 1)


# ---------------------------------------------------------------------------
# CuPy (NVIDIA GPU)
# ---------------------------------------------------------------------------


def csr_device_bytes(mat_csr: Any) -> int:
    """Bytes a SciPy CSR array will occupy on a CuPy device.

    ``cupyx`` stores ``indices`` and ``indptr`` in one shared index dtype, so
    a nonzero count past the int32 limit widens both to 8 bytes per entry.
    """
    nnz = int(mat_csr.data.size)
    int32_max = np.iinfo(np.int32).max
    idx_size = 8 if (max(mat_csr.shape) > int32_max or nnz > int32_max) else 4
    return int(mat_csr.data.nbytes + (nnz + mat_csr.shape[0] + 1) * idx_size)


def _managed_memory_supported() -> bool:
    """Whether this device can oversubscribe its memory."""
    import cupy as cp  # ty: ignore[unresolved-import]

    try:
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    except Exception:  # noqa: BLE001 - a probe; any failure means "no".
        return False
    return bool(props.get("managedMemory")) and bool(
        props.get("concurrentManagedAccess")
    )


def ensure_device_capacity(nbytes: int, what: str = "matrix") -> bool:
    """Install the managed-memory allocator if ``nbytes`` will not fit.

    Raises
    ------
    MemoryError
        If the transfer does not fit and the device cannot oversubscribe, with
        the sizes involved and a pointer at ``PARAFAC2_BACKEND=cpu``.
    """
    global _managed_allocator_installed
    import cupy as cp  # ty: ignore[unresolved-import]

    if _managed_allocator_installed:
        return True

    free, total = cp.cuda.Device().mem_info
    if nbytes <= free * DEVICE_MEMORY_HEADROOM:
        return False

    if not _managed_memory_supported():
        raise MemoryError(
            f"The {what} needs {nbytes / 1e9:.1f} GB on the device but only "
            f"{free / 1e9:.1f} GB of {total / 1e9:.1f} GB is free, and this "
            "device cannot oversubscribe its memory (managed memory "
            "unsupported). Run on the CPU instead, either with "
            f"backend='cpu' or {BACKEND_ENV_VAR}=cpu."
        )

    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
    _managed_allocator_installed = True
    return True


def to_cupy_dense(mat: np.ndarray) -> Any:
    """Move a dense matrix onto the CuPy device."""
    import cupy as cp  # ty: ignore[unresolved-import]

    ensure_device_capacity(int(mat.nbytes), what="data matrix")
    return cp.asarray(mat)


def to_cupy_csr(mat_csr: Any) -> Any:
    """Move a SciPy CSR array onto the CuPy device.

    Switches to managed memory first when the matrix is larger than free
    device memory, so a dataset bigger than the card is paged rather than
    refused.
    """
    import cupy as cp  # ty: ignore[unresolved-import]
    import cupyx.scipy.sparse as cpsparse  # ty: ignore[unresolved-import]

    ensure_device_capacity(csr_device_bytes(mat_csr), what="data matrix")

    return cpsparse.csr_matrix(
        (
            cp.asarray(mat_csr.data),
            cp.asarray(mat_csr.indices),
            cp.asarray(mat_csr.indptr),
        ),
        shape=mat_csr.shape,
    )


def cupy_dense_matmul(
    device_mat: Any, rhs: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Compute ``device_mat @ rhs`` for a dense CuPy-resident matrix.

    Dense products go through cuBLAS's plain ``@``, which has no int32 index
    limitation to work around.
    """
    import cupy as cp  # ty: ignore[unresolved-import]

    return cp.asnumpy(device_mat @ cp.asarray(rhs))


def cupy_dense_rmatmul(
    lhs: np.ndarray, device_mat: Any, shape: tuple[int, int]
) -> np.ndarray:
    """Compute ``lhs @ device_mat`` for a dense CuPy-resident matrix."""
    import cupy as cp  # ty: ignore[unresolved-import]

    return cp.asnumpy(cp.asarray(lhs) @ device_mat)


def cupy_csr_matmul(
    device_mat: Any, rhs: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Compute ``device_mat @ rhs`` (sparse CSR @ dense) via nvmath-python.

    CuPy's own ``@`` for a CSR-by-dense product prefers cuSPARSE's legacy
    ``csrmm2`` routine, which only understands 32-bit indices: it hands the
    raw ``indices``/``indptr`` device pointers straight to that int32-only
    API with no dtype check, so once a matrix has more than ``2**31 - 1``
    nonzeros (and therefore int64 indices), the call reads those buffers as
    the wrong width and silently corrupts the result. nvmath-python's
    ``nvmath.sparse.matmul`` goes through cuSPARSE's *generic* SpMM API
    instead, which is told the operands' actual index type
    (``CUSPARSE_INDEX_64I`` when needed), so it stays correct at any ``nnz``.
    """
    import cupy as cp  # ty: ignore[unresolved-import]
    import nvmath  # ty: ignore[unresolved-import]

    cp_rhs = cp.asarray(rhs)
    rhs_2d = cp_rhs[:, None] if cp_rhs.ndim == 1 else cp_rhs
    out_dtype = cp.result_type(device_mat.dtype, cp_rhs.dtype)
    out = cp.zeros((device_mat.shape[0], rhs_2d.shape[1]), dtype=out_dtype)
    res = nvmath.sparse.matmul(device_mat, rhs_2d, out)
    return cp.asnumpy(res.ravel() if cp_rhs.ndim == 1 else res)


def cupy_csr_rmatmul(
    lhs: np.ndarray, device_mat: Any, shape: tuple[int, int]
) -> np.ndarray:
    """Compute ``lhs @ device_mat`` (dense @ sparse CSR) via nvmath-python.

    nvmath's SpMM always takes the sparse operand first, so this computes
    the equivalent ``(device_mat.T @ lhs.T).T`` using the ``is_transpose``
    matrix qualifier rather than materializing a transposed copy of
    ``device_mat``. See :func:`cupy_csr_matmul` for why cuSPARSE's legacy
    routines (which CuPy's own ``@`` would otherwise use here) are unsafe
    for int64-indexed matrices.
    """
    import cupy as cp  # ty: ignore[unresolved-import]
    import nvmath  # ty: ignore[unresolved-import]

    cp_lhs = cp.asarray(lhs)
    lhs_2d = cp_lhs[None, :] if cp_lhs.ndim == 1 else cp_lhs
    qualifiers = np.zeros(3, dtype=nvmath.sparse.matmul_matrix_qualifiers_dtype)
    qualifiers[0]["is_transpose"] = 1
    out_dtype = cp.result_type(device_mat.dtype, cp_lhs.dtype)
    out = cp.zeros((device_mat.shape[1], lhs_2d.shape[0]), dtype=out_dtype)
    res = nvmath.sparse.matmul(
        device_mat, cp.asfortranarray(lhs_2d.T), out, qualifiers=qualifiers
    ).T
    return cp.asnumpy(res.ravel() if cp_lhs.ndim == 1 else res)


# ---------------------------------------------------------------------------
# MKL (multithreaded CPU sparse kernels)
# ---------------------------------------------------------------------------


def mkl_dot() -> Any:
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
