"""
The duck-typed matrix contract the PARAFAC2 fit talks to, and the wrappers
that give NumPy arrays and SciPy CSR arrays that interface.

Everything downstream of :func:`as_matrix` -- :mod:`parafac2.utils`,
:mod:`parafac2.compress`, :mod:`parafac2.parafac2` -- reaches the raw data
*only* through this contract. Nothing there branches on the data's type.

The contract
------------
An object usable as ``X`` must provide:

``shape``
    The ``(n_rows, n_cols)`` tuple.
``dtype``
    The NumPy dtype products should be taken in. Callers build their dense
    operands in this dtype so the product never has to upcast the (possibly
    enormous) data matrix.
``X @ rhs`` and ``lhs @ X``
    Matrix products against a 1-D or 2-D NumPy array, returning a **float64**
    NumPy array. Any mean-centering the matrix represents must already be
    applied -- ``X @ rhs`` means ``(X_raw - 1 mu^T) @ rhs``.
``norm_sq()``
    The squared Frobenius norm of the (centered) matrix, as a float.
``slice_norms(condition_idxs, n_cond)``
    Per-condition Frobenius norms of the (centered) rows, as a length
    ``n_cond`` array.
``to_device(backend)``
    Return an object satisfying this same contract with its data resident on
    ``backend`` (``'cupy'`` or ``'mlx'``). Only needed to run on a GPU
    backend; parafac2 has no transfer logic of its own for a third-party
    storage format.
``__array_ufunc__ = None``
    Required, so that ``lhs @ X`` for a plain NumPy ``lhs`` defers to ``X``'s
    own ``__rmatmul__`` instead of NumPy trying to broadcast ``X`` into an
    ``ndarray`` (which fails outright for a type that cannot be converted to
    one, as most duck-typed backends cannot).

:func:`as_linear_operator` adapts any of these to a SciPy
:class:`~scipy.sparse.linalg.LinearOperator`, which is how the data reaches
SciPy's decompositions (the randomized SVD in :mod:`parafac2.utils`) without
those ever learning what the data is or where it lives.

:class:`Matrix` implements everything but the two products and the two norms,
so subclassing it is the easy way to satisfy the contract; ``vsparse``-style
types that implement it structurally work just as well and are passed through
:func:`as_matrix` untouched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.sparse import csr_array, issparse
from scipy.sparse.linalg import LinearOperator

from .backend import (
    cupy_csr_matmul,
    cupy_csr_rmatmul,
    cupy_dense_matmul,
    cupy_dense_rmatmul,
    get_backend,
    mkl_dot,
    mlx_csr_matmul,
    mlx_csr_rmatmul,
    mlx_dense_matmul,
    mlx_dense_rmatmul,
    to_cupy_csr,
    to_cupy_dense,
    to_mlx_csr,
    to_mlx_dense,
)


def _clean_means(means: np.ndarray | None) -> np.ndarray | None:
    """Normalize a ``means`` argument, collapsing "no centering" to ``None``."""
    if means is None:
        return None
    means_arr = np.asarray(means, dtype=np.float64).ravel()
    return None if not np.any(means_arr) else means_arr


class Matrix(ABC):
    """Base class implementing the duck-typed matrix contract.

    Subclasses supply the raw products (:meth:`_matmul`/:meth:`_rmatmul`,
    which receive an operand already cast to ``self.dtype`` and made
    C-contiguous) and the two norms. This class layers the shared parts on
    top: operand coercion, the float64 result contract, and the implicit
    mean-centering, which is applied as a rank-1 correction to the (small)
    product rather than by materializing ``X - 1 mu^T``.

    Parameters
    ----------
    shape : tuple[int, int]
        The matrix's ``(n_rows, n_cols)`` shape.
    dtype : np.dtype
        The dtype products are taken in.
    means : np.ndarray | None, default None
        Per-column means to subtract implicitly. ``None`` or an all-zero
        array means no centering.
    """

    __array_priority__ = 1000
    __array_ufunc__ = None

    def __init__(
        self,
        shape: tuple[int, int],
        dtype: Any,
        means: np.ndarray | None = None,
    ) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self.dtype = np.dtype(dtype)
        self.means = _clean_means(means)

    @abstractmethod
    def _matmul(self, rhs: np.ndarray) -> np.ndarray:
        """Compute the uncentered ``self @ rhs``."""

    @abstractmethod
    def _rmatmul(self, lhs: np.ndarray) -> np.ndarray:
        """Compute the uncentered ``lhs @ self``."""

    @abstractmethod
    def norm_sq(self) -> float:
        """Return the squared Frobenius norm of the (centered) matrix."""

    @abstractmethod
    def slice_norms(self, condition_idxs: np.ndarray, n_cond: int) -> np.ndarray:
        """Return the per-condition Frobenius norms of the (centered) rows.

        Parameters
        ----------
        condition_idxs : np.ndarray
            Integer array assigning each row to a condition in ``[0, n_cond)``.
        n_cond : int
            The total number of conditions.
        """

    def to_device(self, backend: str | None = None) -> Matrix:
        """Return this matrix with its data resident on ``backend``."""
        raise NotImplementedError

    def _operand(self, arr: np.ndarray) -> np.ndarray:
        """Cast a dense operand to this matrix's dtype and C-contiguous layout.

        Taking the product in ``X``'s own dtype matters: handing a float64
        operand to a float32 sparse ``X`` makes SciPy upcast the *whole*
        sparse matrix, doubling both the memory traffic that dominates the
        fit and the peak memory.
        """
        return np.ascontiguousarray(arr, dtype=self.dtype)

    def __matmul__(self, rhs: np.ndarray) -> np.ndarray:
        """Compute ``(self - 1 mu^T) @ rhs`` as a float64 NumPy array."""
        rhs = np.asarray(rhs)
        prod = np.asarray(self._matmul(self._operand(rhs)), dtype=np.float64)
        if self.means is not None:
            prod -= self.means @ rhs
        return prod

    def __rmatmul__(self, lhs: np.ndarray) -> np.ndarray:
        """Compute ``lhs @ (self - 1 mu^T)`` as a float64 NumPy array."""
        lhs = np.asarray(lhs)
        prod = np.asarray(self._rmatmul(self._operand(lhs)), dtype=np.float64)
        if self.means is not None:
            if lhs.ndim == 1:
                prod -= lhs.sum() * self.means
            else:
                prod -= np.outer(lhs.sum(axis=1), self.means)
        return prod


class DenseMatrix(Matrix):
    """A dense NumPy array with optional implicit mean-centering.

    Parameters
    ----------
    mat : np.ndarray
        The dense data matrix of shape ``(n_rows, n_cols)``.
    means : np.ndarray | None, default None
        Per-column means to subtract implicitly.
    """

    def __init__(self, mat: np.ndarray, means: np.ndarray | None = None) -> None:
        self.mat = np.asarray(mat)
        super().__init__(self.mat.shape, self.mat.dtype, means)

    def _matmul(self, rhs: np.ndarray) -> np.ndarray:
        return self.mat @ rhs

    def _rmatmul(self, lhs: np.ndarray) -> np.ndarray:
        return lhs @ self.mat

    def norm_sq(self) -> float:
        if self.means is None:
            return float(np.sum(self.mat**2))
        return float(np.sum((self.mat - self.means) ** 2))

    def slice_norms(self, condition_idxs: np.ndarray, n_cond: int) -> np.ndarray:
        idxs = np.asarray(condition_idxs)
        centered = self.mat if self.means is None else self.mat - self.means
        row_sums_sq = np.sum(centered**2, axis=1)
        return np.sqrt(np.bincount(idxs, weights=row_sums_sq, minlength=n_cond))

    def to_device(self, backend: str | None = None) -> Matrix:
        chosen = get_backend(backend)
        if chosen == "cupy":
            return DeviceMatrix(
                self, to_cupy_dense(self.mat), cupy_dense_matmul, cupy_dense_rmatmul
            )
        if chosen == "mlx":
            return DeviceMatrix(
                self, to_mlx_dense(self.mat), mlx_dense_matmul, mlx_dense_rmatmul
            )
        return self


class CSRMatrix(Matrix):
    """A SciPy CSR array with optional implicit mean-centering.

    The norms expand ``||X - 1 mu^T||^2`` into sum-of-squares and cross terms
    over the stored nonzeros, so a centered norm never densifies ``X``.

    Parameters
    ----------
    mat : csr_array
        The sparse data matrix of shape ``(n_rows, n_cols)``. Any SciPy
        sparse format is accepted and converted to CSR.
    means : np.ndarray | None, default None
        Per-column means to subtract implicitly.
    """

    def __init__(self, mat: Any, means: np.ndarray | None = None) -> None:
        self.mat = mat if isinstance(mat, csr_array) else csr_array(mat)
        super().__init__(self.mat.shape, self.mat.dtype, means)

    def _mkl_dot(self) -> Any:
        """MKL's kernel if it is installed and it supports this dtype.

        MKL needs single- or double-precision data; the operand's dtype and
        layout are already guaranteed by :meth:`Matrix._operand`.
        """
        if self.dtype not in (np.float32, np.float64):
            return None
        return mkl_dot()

    def _matmul(self, rhs: np.ndarray) -> np.ndarray:
        dot = self._mkl_dot()
        return self.mat @ rhs if dot is None else dot(self.mat, rhs)

    def _rmatmul(self, lhs: np.ndarray) -> np.ndarray:
        dot = self._mkl_dot()
        return lhs @ self.mat if dot is None else dot(lhs, self.mat)

    def norm_sq(self) -> float:
        data = self.mat.data
        if self.means is None:
            return float(np.sum(data**2))
        term1 = np.sum(data**2)
        term2 = -2.0 * np.sum(data * self.means[self.mat.indices])
        term3 = self.shape[0] * np.sum(self.means**2)
        return float(term1 + term2 + term3)

    def slice_norms(self, condition_idxs: np.ndarray, n_cond: int) -> np.ndarray:
        idxs = np.asarray(condition_idxs)
        group_of_nnz = np.repeat(idxs, np.diff(self.mat.indptr))
        data = self.mat.data.astype(np.float64)
        sums_sq = np.bincount(group_of_nnz, weights=data**2, minlength=n_cond)
        if self.means is None:
            return np.sqrt(sums_sq)

        counts = np.bincount(idxs, minlength=n_cond).astype(np.float64)
        cross = np.bincount(
            group_of_nnz,
            weights=data * self.means[self.mat.indices],
            minlength=n_cond,
        )
        mean_sq_total = np.sum(self.means**2)
        return np.sqrt(np.maximum(sums_sq - 2.0 * cross + counts * mean_sq_total, 0.0))

    def to_device(self, backend: str | None = None) -> Matrix:
        chosen = get_backend(backend)
        if chosen == "cupy":
            return DeviceMatrix(
                self, to_cupy_csr(self.mat), cupy_csr_matmul, cupy_csr_rmatmul
            )
        if chosen == "mlx":
            return DeviceMatrix(
                self, to_mlx_csr(self.mat), mlx_csr_matmul, mlx_csr_rmatmul
            )
        return self


class DeviceMatrix(Matrix):
    """A host matrix whose data has been transferred to an accelerator.

    Holds the device-resident payload plus the product routines that
    understand it, and delegates the norms back to the host matrix (they are
    computed once at setup, from data the host object already owns).

    Parameters
    ----------
    host : Matrix
        The CPU-resident matrix this was transferred from.
    device_mat : Any
        The device-resident payload (a CuPy array or sparse matrix, an MLX
        array, or the ``(data, indices, indptr)`` MLX tuple for CSR).
    matmul_fn, rmatmul_fn : callable
        The backend routines computing ``device_mat @ rhs`` and
        ``lhs @ device_mat``, each taking the logical shape as a third
        argument.
    """

    def __init__(
        self,
        host: Matrix,
        device_mat: Any,
        matmul_fn: Any,
        rmatmul_fn: Any,
    ) -> None:
        self.host = host
        self.device_mat = device_mat
        self._matmul_fn = matmul_fn
        self._rmatmul_fn = rmatmul_fn
        super().__init__(host.shape, host.dtype, host.means)

    def _matmul(self, rhs: np.ndarray) -> np.ndarray:
        return self._matmul_fn(self.device_mat, rhs, self.shape)

    def _rmatmul(self, lhs: np.ndarray) -> np.ndarray:
        return self._rmatmul_fn(lhs, self.device_mat, self.shape)

    def norm_sq(self) -> float:
        return self.host.norm_sq()

    def slice_norms(self, condition_idxs: np.ndarray, n_cond: int) -> np.ndarray:
        return self.host.slice_norms(condition_idxs, n_cond)

    def to_device(self, backend: str | None = None) -> Matrix:
        return self.host.to_device(backend)


def as_matrix(mat: Any, means: np.ndarray | None = None) -> Any:
    """Return ``mat`` as something satisfying the duck-typed matrix contract.

    This is the single place in the package that looks at the data's type:
    a NumPy array or SciPy sparse matrix is wrapped, and anything else is
    assumed to implement the contract itself and passed through untouched.

    Parameters
    ----------
    mat : Any
        A NumPy array, a SciPy sparse matrix, or an object already
        implementing the contract (including a :class:`Matrix`).
    means : np.ndarray | None, default None
        Per-column means to center by. Only meaningful for a plain NumPy or
        SciPy input: a type that implements the contract itself already
        accounts for its own centering.

    Returns
    -------
    Any
        A :class:`DenseMatrix` or :class:`CSRMatrix` for NumPy/SciPy input,
        otherwise ``mat`` itself.

    Raises
    ------
    ValueError
        If a nonzero ``means`` is passed alongside a matrix that already
        handles its own centering.
    """
    if isinstance(mat, np.ndarray):
        return DenseMatrix(mat, means)
    if issparse(mat):
        return CSRMatrix(mat, means)

    if _clean_means(means) is not None:
        raise ValueError(
            f"{type(mat).__name__} implements its own centering, so passing a "
            "nonzero `means` alongside it is not supported."
        )
    return mat


class _MatrixOperator(LinearOperator):
    """A duck-typed matrix seen as a SciPy linear operator.

    A pure adapter: every product is forwarded to the contract, so this
    inherits whatever the matrix already does about mean-centering, dtype
    handling, and -- for a device-resident matrix -- moving the dense
    multiplicand to wherever the data lives. ``matmat``/``rmatmat`` forward a
    multi-column operand whole, so a block product costs one pass over the
    data rather than one pass per column.
    """

    def __init__(self, X: Any) -> None:
        # Always float64: that is what the contract's products return, and
        # what SciPy's decompositions require. The product itself is still
        # taken in the data's own dtype, so the data is never upcast.
        super().__init__(np.dtype(np.float64), X.shape)
        self.X = X

    def _matmat(self, rhs: np.ndarray) -> np.ndarray:
        return np.asarray(self.X @ rhs, dtype=np.float64)

    def _rmatmat(self, lhs: np.ndarray) -> np.ndarray:
        return np.asarray(lhs.T @ self.X, dtype=np.float64).T

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.X @ np.ravel(x), dtype=np.float64)

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(np.ravel(x) @ self.X, dtype=np.float64)


def as_linear_operator(mat: Any, means: np.ndarray | None = None) -> LinearOperator:
    """Return ``mat`` as a SciPy :class:`~scipy.sparse.linalg.LinearOperator`.

    This is how the data is handed to SciPy's decompositions (see
    :func:`~parafac2.utils.randomized_svd_right`): anything :func:`as_matrix`
    accepts becomes an operator without the decomposition ever learning what
    the data actually is or where it lives.

    Parameters
    ----------
    mat : Any
        Anything :func:`as_matrix` accepts.
    means : np.ndarray | None, default None
        Per-column means to center by, as accepted by :func:`as_matrix`.

    Returns
    -------
    LinearOperator
        An ``(n_rows, n_cols)`` float64 operator applying ``X @ rhs`` and
        ``lhs @ X``.
    """
    return _MatrixOperator(as_matrix(mat, means))


def to_gpu(
    mat: Any, backend: str | None = None, means: np.ndarray | None = None
) -> Any:
    """Wrap ``mat`` for the fit and move it to the resolved backend's device.

    Parameters
    ----------
    mat : Any
        The matrix to (optionally) transfer, as accepted by :func:`as_matrix`.
    backend : str, optional
        One of ``'mlx'``, ``'cupy'``, or ``'cpu'``. If ``None``, the first
        available accelerator is auto-detected (see
        :func:`~parafac2.backend.get_backend`).
    means : np.ndarray | None, default None
        Per-column means to center by, as accepted by :func:`as_matrix`.

    Returns
    -------
    Any
        The device-resident matrix on a GPU backend, otherwise the
        CPU-resident one.

    Raises
    ------
    TypeError
        If a GPU backend is selected for a duck-typed matrix that does not
        implement ``to_device``.
    """
    X = as_matrix(mat, means)
    chosen = get_backend(backend)
    if chosen == "cpu":
        return X

    to_device = getattr(X, "to_device", None)
    if to_device is None:
        raise TypeError(
            f"{type(X).__name__} does not implement `to_device`, so it cannot "
            f"run on the {chosen!r} backend. Pass backend='cpu', or convert "
            "it to a NumPy array or SciPy sparse array first."
        )
    return to_device(chosen)


def GPUMatrix(
    mat: Any, backend: str | None = None, means: np.ndarray | None = None
) -> Any:
    """Deprecated alias for :func:`to_gpu`, kept for backwards compatibility.

    The GPU-resident matrix is now just another implementation of the
    duck-typed contract (see :class:`DeviceMatrix`) rather than a separate
    wrapper class.
    """
    return to_gpu(mat, backend=backend, means=means)
