from typing import TYPE_CHECKING, cast

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
    l1_c: float = 0.0,
    max_iter_cd: int = 100,
    tol_cd: float = 1e-5,
    orth_b: bool = False,
) -> list[np.ndarray]:
    """
    Perform sequential PARAFAC updates for all modes using pre-computed MTTKRPs.
    This corresponds to Option 2: Sequential with reuse.

    All factors here are rank-sized (n_cond, rank), (rank, rank), or
    (n_genes, rank).
    """
    if mode == 1 and orth_b:
        u, _, vh = np.linalg.svd(mttkrp, full_matrices=False)
        factors[1] = u @ vh
        return factors

    rank = factors[0].shape[1]

    # Compute Gram matrix product using current factors
    v = np.ones((rank, rank))
    for i, factor in enumerate(factors):
        if i != mode:
            v *= factor.T @ factor

    # Update the factor for the current mode
    if mode == 2 and l1_c > 0.0:
        C = factors[2].copy()
        M = mttkrp
        for _ in range(max_iter_cd):
            C_old = C.copy()
            for j in range(rank):
                rho_j = M[:, j] - (C @ v[j, :]) + v[j, j] * C[:, j]
                denom = v[j, j]
                if denom > 1e-15:
                    # Threshold the unconstrained least-squares solution
                    # (rho_j / denom) directly, rather than thresholding the
                    # raw residual rho_j before dividing by denom. denom is
                    # component j's Gram-diagonal "energy"
                    # (~ ||A_j||^2 * ||B_j||^2), which varies a lot across
                    # components; thresholding pre-division makes the
                    # effective penalty in C's own units equal to
                    # l1_c / denom, so low-energy components get
                    # disproportionately shrunk and can collapse to all-zero
                    # (and, once one column is fully zero, its Gram
                    # contribution can make the next factor's solve
                    # singular) even at moderate l1_c. Thresholding
                    # post-division makes l1_c a column-scale-invariant
                    # penalty on C itself.
                    c_ls = rho_j / denom
                    C[:, j] = np.sign(c_ls) * np.maximum(0.0, np.abs(c_ls) - l1_c)
                else:
                    C[:, j] = np.zeros_like(rho_j)
            if np.max(np.abs(C - C_old)) < tol_cd:
                break
        factors[2] = C
    else:
        try:
            factors[mode] = np.linalg.solve(v.T, mttkrp.T).T
        except np.linalg.LinAlgError:
            # v can be exactly singular when l1_c has thresholded an entire
            # column of C to zero (a legitimate outcome of L1 regularization
            # -- that component is pruned): the corresponding diagonal of the
            # Hadamard-product Gram v is then zero, and since v is PSD a zero
            # diagonal entry forces the whole row/column to zero too. Fall
            # back to the minimum-norm least-squares solution (via SVD)
            # instead of crashing; the dead component's row comes out ~0,
            # which is the correct behavior, and the fit can recover it on a
            # later iteration if l1_c's threshold is no longer binding there.
            factors[mode] = np.linalg.lstsq(v.T, mttkrp.T, rcond=None)[0].T

    return factors


def project_data(
    X: "np.ndarray | csr_array",
    condition_unique_idxs: np.ndarray,
    means: np.ndarray | None,
    factors: list[np.ndarray],
    norm_X_sq: float,
    mode: int,
    return_projections: bool = False,
    cond_indices: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, float] | list[np.ndarray]:
    """
    Project each condition's data onto the current factors and accumulate the
    MTTKRP for the requested mode.
    """
    A, B, C = factors
    rank = B.shape[0]
    n_cond = A.shape[0]
    CtC = C.T @ C
    BtB = B.T @ B
    norm_sq_err = norm_X_sq

    # Single GEMM for W = (X - 1 mu^T) @ C = X @ C - 1 (mu @ C)
    W = X @ C
    if means is not None:
        W = W - (means @ C)

    if cond_indices is None:
        cond_indices = [
            np.flatnonzero(condition_unique_idxs == i) for i in range(n_cond)
        ]

    if mode == 0:
        mttkrp = np.zeros((n_cond, rank))
    elif mode == 1:
        mttkrp = np.zeros((rank, rank))
    else:
        H = np.empty((X.shape[0], rank), dtype=np.float64)

    proj_list = []
    for i in range(n_cond):
        idx_i = cond_indices[i]
        W_i = W[idx_i]
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
        B_i_inner = A[i][:, np.newaxis] * BtB * A[i]

        if mode == 0:
            mttkrp[i] = np.sum(psc * B, axis=0)
        elif mode == 1:
            mttkrp += psc * A[i]
        else:
            # Mode 2 updates C
            H[idx_i] = proj @ (B * A[i])

        norm_sq_err -= 2.0 * np.einsum("r,jr,jr->", A[i], B, psc)
        norm_sq_err += (B_i_inner * CtC).sum()

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
    # Order components by condition variance. A component pruned entirely to
    # zero by l1_c regularization (dead in every factor) has mean == 0 here;
    # treat it as minimum variance rather than propagating a 0/0 NaN.
    col_mean = np.mean(factors[0], axis=0)
    gini = np.divide(
        np.var(factors[0], axis=0),
        col_mean,
        out=np.zeros_like(col_mean),
        where=col_mean != 0,
    )
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
