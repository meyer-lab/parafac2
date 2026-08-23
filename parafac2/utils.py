from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse
from tensorly.cp_tensor import cp_flip_sign, cp_normalize

if TYPE_CHECKING:
    from scipy.sparse import csr_array


def calc_norm_sq(X: "np.ndarray | csr_array", means: np.ndarray | None = None) -> float:
    """Return the squared Frobenius norm of the mean-centered matrix."""
    if means is None or np.all(means == 0):
        if issparse(X):
            return float(np.sum(X.data**2))
        return float(np.sum(X**2))

    means_arr = np.asarray(means).ravel()
    if issparse(X):
        mat_csr = cast("csr_array", X)
        M = mat_csr.shape[0]
        term1 = np.sum(mat_csr.data**2)
        term2 = -2.0 * np.sum(mat_csr.data * means_arr[mat_csr.indices])
        term3 = M * np.sum(means_arr**2)
        return float(term1 + term2 + term3)
    return float(np.sum((X - means_arr) ** 2))


def parafac_update(
    factors: list[np.ndarray],
    mttkrp: np.ndarray,
    mode: int,
) -> list[np.ndarray]:
    """
    Perform PARAFAC update for the requested mode using pre-computed MTTKRP.
    """
    rank = factors[0].shape[1]

    # Compute Gram matrix product using current factors
    v = np.ones((rank, rank))
    for i, factor in enumerate(factors):
        if i != mode:
            v *= factor.T @ factor

    try:
        factors[mode] = np.linalg.solve(v.T, mttkrp.T).T
    except np.linalg.LinAlgError:
        factors[mode] = np.linalg.lstsq(v.T, mttkrp.T, rcond=None)[0].T

    return factors


def project_data(
    X: Any,
    condition_unique_idxs: np.ndarray,
    means: np.ndarray | None,
    factors: list[np.ndarray],
    norm_X_sq: float,
    mode: int,
    return_projections: bool = False,
) -> tuple[np.ndarray, float] | list[np.ndarray]:
    """
    Project each condition's data onto the current factors and accumulate the
    MTTKRP for the requested mode.
    """
    A, B, C = factors
    rank = B.shape[0]
    n_cond = A.shape[0]

    # Initialize error with full tensor contraction ||X||^2 + Tr(A^T A * B^T B * C^T C)
    norm_sq_err = norm_X_sq + float(((A.T @ A) * (B.T @ B) * (C.T @ C)).sum())

    # Single GEMM for W = (X - 1 mu^T) @ C = X @ C - 1 (mu @ C)
    W = X @ C
    if means is not None:
        W = W - (means @ C)

    if mode == 0:
        mttkrp = np.zeros((n_cond, rank))
    elif mode == 1:
        mttkrp = np.zeros((rank, rank))
    else:
        H = np.empty((X.shape[0], rank), dtype=np.float64)

    proj_list = []
    for i in range(n_cond):
        cond_i = condition_unique_idxs == i
        W_i = W[cond_i]
        T_i = (B * A[i]).T  # (rank, rank)
        M = W_i @ T_i
        G = M.T @ M  # (rank, rank)
        _, V = np.linalg.eigh(G)
        MV = M @ V  # ≈ U @ S @ D, orthogonal columns
        col_norms = np.linalg.norm(MV, axis=0, keepdims=True)
        safe_norms = np.where(col_norms > 1e-10, col_norms, 1.0)
        proj = (MV / safe_norms) @ V.T  # D cancels -> U @ Vh
        proj_list.append(proj)

        if return_projections:
            continue

        psc = proj.T @ W_i  # (rank, rank) dense product
        m_i = np.sum(psc * B, axis=0)
        norm_sq_err -= 2.0 * float(np.dot(A[i], m_i))

        if mode == 0:
            mttkrp[i] = m_i
        elif mode == 1:
            mttkrp += psc * A[i]
        else:
            # Mode 2 updates C
            H[cond_i] = proj @ (B * A[i])

    if return_projections:
        return proj_list

    if mode == 2:
        mttkrp = (H.T @ X).T
        if means is not None:
            mttkrp -= np.outer(means, np.sum(H, axis=0))

    return mttkrp, float(norm_sq_err)


def standardize_pf2(
    factors: list[np.ndarray], projections: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    # Order components by condition variance-to-mean ratio
    gini = np.var(factors[0], axis=0) / np.mean(factors[0], axis=0)
    gini_idx = np.argsort(gini)
    factors = [f[:, gini_idx] for f in factors]

    weights, factors = cp_flip_sign(cp_normalize((None, factors)), mode=1)

    # Order eigen-cells to maximize the diagonal of B
    _, col_ind = linear_sum_assignment(np.abs(factors[1].T), maximize=True)
    factors[1] = factors[1][col_ind, :]
    projections = [p[:, col_ind] for p in projections]

    # Flip the sign based on B
    signn = np.sign(np.diag(factors[1]))
    factors[1] *= signn[:, np.newaxis]
    projections = [p * signn for p in projections]

    return weights, factors, projections
