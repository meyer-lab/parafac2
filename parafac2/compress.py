"""
CANDELINC compression for PARAFAC2.

Compresses the gene and cell modes of single-cell datasets prior to
PARAFAC2 factorization, collapsing the problem to small dense per-condition
cores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from .backend import to_gpu
from .utils import (
    calc_W,
    extract_dataset_info,
    polar_factor,
    randomized_svd_right,
)

if TYPE_CHECKING:
    import anndata


@dataclass
class CompressedData:
    """Compressed representation of a multi-condition single-cell dataset.

    Stores the small dense per-condition cores and the orthonormal bases for
    the gene mode (``Q``) and cell mode (``Q_k``), enabling fast rank sweeps
    without touching the raw data again.
    """

    cores: list[np.ndarray]
    """List of length ``n_cond`` with dense cores ``Y_k`` of shape ``(L_c_k, L_g)``."""

    Q: np.ndarray
    """Gene projection basis of shape ``(n_genes, L_g)`` with orthonormal columns."""

    Q_k: list[np.ndarray | None] | None
    """Per-condition cell projection bases ``(n_cells_k, L_c_k)`` or ``None``."""

    condition_unique_idxs: np.ndarray
    """Integer condition indices for each cell."""

    norm_tensor: float
    """Total squared Frobenius norm of the original mean-centered dataset."""

    lost_var: float
    """Variance discarded by the compression projectors."""

    total_cells: int
    """Total number of cells across all conditions."""

    n_genes: int
    """Number of genes in the uncompressed dataset."""

    n_cond: int
    """Number of conditions."""

    slice_weights: np.ndarray | None = None
    """Optional per-condition slice weights for normalized ALS."""

    means: np.ndarray | None = None
    """Per-gene means subtracted during compression."""

    adata: anndata.AnnData | None = None
    """Reference to original AnnData object if available."""

    @property
    def L_g(self) -> int:
        """Gene compression dimension."""
        return self.Q.shape[1]

    @property
    def max_cell_dim(self) -> int:
        """Maximum cell dimension across cores."""
        return max(c.shape[0] for c in self.cores)


def compress_genes(
    X: Any,
    means: np.ndarray | None,
    L_g: int,
    n_power_iter: int = 2,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute gene-mode compression projector Q and compressed matrix Xc.

    Parameters
    ----------
    X : Any
        Stacked data matrix of shape ``(total_cells, n_genes)``.
    means : np.ndarray | None
        Per-gene means for centering.
    L_g : int
        Target gene subspace dimension.
    n_power_iter : int, default 2
        Number of power iterations for randomized SVD.
    random_state : int | np.random.Generator | None, default None
        Random seed or generator.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        ``(X_c, Q, norm_Xc_sq)`` where ``X_c = (X - 1 mu^T) @ Q`` of shape
        ``(total_cells, L_g)``, ``Q`` is ``(n_genes, L_g)`` orthonormal, and
        ``norm_Xc_sq`` is the squared Frobenius norm of ``X_c``.
    """
    _n_cells, n_genes = X.shape
    L_g = min(n_genes, L_g)

    Q = randomized_svd_right(
        X,
        means,
        n_components=L_g,
        n_oversamples=0,
        n_power_iter=n_power_iter,
        random_state=random_state,
    )
    X_c = calc_W(X, means, Q)
    norm_Xc_sq = float(np.sum(X_c**2))

    return X_c, Q, norm_Xc_sq


def compress_cells(
    X_c: np.ndarray,
    condition_unique_idxs: np.ndarray,
    L_c: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray | None] | None, float]:
    """Compute per-condition cell-mode compression projectors Q_k and cores Y_k.

    Parameters
    ----------
    X_c : np.ndarray
        Gene-compressed data matrix of shape ``(total_cells, L_g)``.
    condition_unique_idxs : np.ndarray
        Integer condition index for each cell.
    L_c : int | None, default None
        Target cell subspace dimension per condition. If ``None``, cell
        compression is skipped (cores are ``X_c`` slices).

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray | None] | None, float]
        ``(cores, Q_k_list, norm_cores_sq)`` where each core ``Y_k`` has
        shape ``(L_c_k, L_g)`` and ``norm_cores_sq`` is the total squared
        Frobenius norm of all cores.
    """
    n_cond = int(np.amax(condition_unique_idxs)) + 1
    cores: list[np.ndarray] = []
    Q_k_list: list[np.ndarray | None] | None = [] if L_c is not None else None
    norm_cores_sq = 0.0

    for i in range(n_cond):
        cond_i = condition_unique_idxs == i
        X_c_i = X_c[cond_i]
        n_k = X_c_i.shape[0]

        if L_c is None or n_k <= L_c:
            cores.append(X_c_i)
            if Q_k_list is not None:
                Q_k_list.append(None)
            norm_cores_sq += float(np.sum(X_c_i**2))
        else:
            # Thin SVD of (n_k, L_g) where L_g <= 100
            U_i, S_i, Vh_i = np.linalg.svd(X_c_i, full_matrices=False)
            L_k = min(n_k, L_c)
            Q_i = U_i[:, :L_k].astype(np.float64)
            Y_i = (S_i[:L_k, np.newaxis] * Vh_i[:L_k, :]).astype(np.float64)
            cores.append(Y_i)
            if Q_k_list is not None:
                Q_k_list.append(Q_i)
            norm_cores_sq += float(np.sum(S_i[:L_k] ** 2))

    return cores, Q_k_list, norm_cores_sq


def compress_dataset(
    X_in: anndata.AnnData,
    L: int | tuple[int, int | None] | str = "auto",
    rank: int | None = None,
    n_power_iter: int = 2,
    random_state: int | np.random.Generator | None = None,
    normalize_slices: bool = False,
    backend: str | None = None,
) -> CompressedData:
    """Compress an AnnData dataset in gene and cell modes.

    Parameters
    ----------
    X_in : anndata.AnnData
        Input dataset with data in ``X_in.X``, condition indices in
        ``X_in.obs["condition_unique_idxs"]``, and optional means in
        ``X_in.var["means"]``.
    L : int | tuple[int, int | None] | str, default "auto"
        Compression dimension(s). If ``"auto"``, picks dimensions based on
        ``rank`` (or default rank 30 if ``rank`` is None). If an int, sets
        both ``L_g = L`` and ``L_c = L``. If a tuple ``(L_g, L_c)``, sets
        gene and cell dimensions individually (pass ``L_c=None`` for
        gene-only compression).
    rank : int | None, default None
        Expected maximum rank to fit on the compressed data. Used when
        ``L="auto"``.
    n_power_iter : int, default 2
        Number of power iterations for randomized SVD.
    random_state : int | np.random.Generator | None, default None
        Random seed or generator.
    normalize_slices : bool, default False
        Whether to precalculate slice weights for normalized ALS.
    backend : str | None, default None
        Compute backend for raw matrix products.

    Returns
    -------
    CompressedData
        The compressed dataset ready for fast PARAFAC2 fitting.
    """
    (
        X_mat,
        condition_unique_idxs,
        means,
        norm_tensor,
        slice_weights,
    ) = extract_dataset_info(X_in, normalize_slices=normalize_slices)
    total_cells, n_genes = X_mat.shape
    n_cond = int(np.amax(condition_unique_idxs)) + 1

    # Determine L_g and L_c
    target_rank = rank if rank is not None else 30
    if isinstance(L, str) and L == "auto":
        L_g_val = min(n_genes, max(4 * target_rank, target_rank + 20))
        L_c_val: int | None = max(4 * target_rank, target_rank + 20)
    elif isinstance(L, tuple):
        L_g_val, L_c_val = L
        L_g_val = min(n_genes, L_g_val)
    elif isinstance(L, (int, np.integer)):
        L_g_val = min(n_genes, int(L))
        L_c_val = int(L)
    else:
        raise ValueError(f"Invalid compression parameter L: {L}")

    X_raw = to_gpu(X_mat, backend=backend)
    X_c, Q, _norm_Xc_sq = compress_genes(
        X_raw,
        means,
        L_g=L_g_val,
        n_power_iter=n_power_iter,
        random_state=random_state,
    )

    cores, Q_k, norm_cores_sq = compress_cells(
        X_c,
        condition_unique_idxs,
        L_c=L_c_val,
    )

    lost_var = float(np.maximum(0.0, norm_tensor - norm_cores_sq))

    return CompressedData(
        cores=cores,
        Q=Q,
        Q_k=Q_k,
        condition_unique_idxs=condition_unique_idxs,
        norm_tensor=norm_tensor,
        lost_var=lost_var,
        total_cells=total_cells,
        n_genes=n_genes,
        n_cond=n_cond,
        slice_weights=slice_weights,
        means=means,
        adata=X_in,
    )


@overload
def project_data_compressed(
    cores: list[np.ndarray],
    factors: list[np.ndarray],
    norm_tensor: float,
    mode: int,
    return_projections: Literal[False] = False,
    slice_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float]: ...


@overload
def project_data_compressed(
    cores: list[np.ndarray],
    factors: list[np.ndarray],
    norm_tensor: float,
    mode: int,
    return_projections: Literal[True],
    slice_weights: np.ndarray | None = None,
) -> list[np.ndarray]: ...


def project_data_compressed(
    cores: list[np.ndarray],
    factors: list[np.ndarray],
    norm_tensor: float,
    mode: int,
    return_projections: bool = False,
    slice_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float] | list[np.ndarray]:
    """Project compressed per-condition cores and accumulate MTTKRP and error.

    Parameters
    ----------
    cores : list[np.ndarray]
        List of per-condition core matrices ``Y_k`` of shape ``(L_c_k, L_g)``.
    factors : list[np.ndarray]
        Current factor matrices ``[A, B, C_L]`` in the compressed space.
    norm_tensor : float
        Squared Frobenius norm of the original mean-centered tensor.
    mode : int
        Mode to update (0, 1, or 2).
    return_projections : bool, default False
        Whether to return the list of projection matrices ``P_tilde_k``.
    slice_weights : np.ndarray | None, default None
        Optional per-condition slice weights.

    Returns
    -------
    tuple[np.ndarray, float] | list[np.ndarray]
        ``(mttkrp, norm_sq_err)`` or list of ``P_tilde_k``.
    """
    A, B, C_L = factors
    rank = B.shape[0]
    n_cond = len(cores)

    norm_sq_err = norm_tensor + float(((A.T @ A) * (B.T @ B) * (C_L.T @ C_L)).sum())

    if mode == 0:
        mttkrp = np.zeros((n_cond, rank), dtype=np.float64)
    elif mode == 1:
        mttkrp = np.zeros((rank, rank), dtype=np.float64)
    else:
        mttkrp = np.zeros_like(C_L, dtype=np.float64)

    proj_list = []
    for i in range(n_cond):
        Y_i = cores[i]
        W_i = Y_i @ C_L  # (L_c_i, rank)
        M = W_i @ (B * A[i]).T  # (L_c_i, rank)
        proj = polar_factor(M)
        proj_list.append(proj)

        if return_projections:
            continue

        psc = proj.T @ W_i  # (rank, rank)
        m_i = np.sum(psc * B, axis=0)
        norm_sq_err -= 2.0 * float(np.dot(A[i], m_i))

        w_i = 1.0 if slice_weights is None else slice_weights[i]

        if mode == 0:
            mttkrp[i] = m_i * w_i
        elif mode == 1:
            mttkrp += psc * A[i] * w_i
        else:
            H_tilde_i = proj @ (B * A[i]) * w_i
            mttkrp += Y_i.T @ H_tilde_i

    if return_projections:
        return proj_list

    return mttkrp, float(norm_sq_err)


def init_compressed_factors(
    cores: list[np.ndarray],
    rank: int,
    random_state: int | np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Initialize factor matrices [A, B, C_L] directly on compressed cores."""
    n_cond = len(cores)
    L_g = cores[0].shape[1]
    assert rank <= L_g, f"Rank {rank} exceeds compressed gene dimension {L_g}"

    # SVD of stacked cores to initialize C_L
    Y_stacked = np.concatenate(cores, axis=0)
    _, _, vh = np.linalg.svd(Y_stacked, full_matrices=False)
    C_L = vh[:rank, :].T.astype(np.float64)

    return [
        np.ones((n_cond, rank), dtype=np.float64),
        np.eye(rank, dtype=np.float64),
        C_L,
    ]
