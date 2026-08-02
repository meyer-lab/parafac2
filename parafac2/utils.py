from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import anndata
import numpy as np
from scipy.optimize import linear_sum_assignment
from tensorly.cp_tensor import cp_flip_sign, cp_normalize

if TYPE_CHECKING:
    from scipy.sparse import csr_array

from .sample import SampleArray


def parafac_update(
    factors: list[np.ndarray],
    mttkrp: np.ndarray,
    mode: int,
    l1_c: float = 0.0,
    max_iter_cd: int = 100,
    tol_cd: float = 1e-5,
) -> list[np.ndarray]:
    """
    Perform sequential PARAFAC updates for all modes using pre-computed MTTKRPs.
    This corresponds to Option 2: Sequential with reuse.

    All factors here are rank-sized (n_cond, rank), (rank, rank), or
    (n_genes, rank).
    """
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
                    soft_val = np.maximum(0.0, np.abs(rho_j) - l1_c)
                    C[:, j] = np.sign(rho_j) * soft_val / denom
                else:
                    C[:, j] = np.zeros_like(rho_j)
            if np.max(np.abs(C - C_old)) < tol_cd:
                break
        factors[2] = C
    else:
        factors[mode] = np.linalg.solve(v.T, mttkrp.T).T

    return factors


def anndata_to_list(X_in: anndata.AnnData) -> list[SampleArray]:
    assert X_in.X is not None
    X_mat = cast("np.ndarray | csr_array", X_in.X)
    sgIndex = cast("np.ndarray", X_in.obs["condition_unique_idxs"].to_numpy(dtype=int))

    if "means" in X_in.var:
        means = X_in.var["means"].to_numpy()
    else:
        means = np.zeros(X_in.shape[1])

    X_list = []
    for i in range(np.amax(sgIndex) + 1):
        slice_mat = X_mat[sgIndex == i]
        X_list.append(SampleArray(slice_mat, means))

    return X_list


def project_data(
    X_list: Sequence[SampleArray],
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
    CtC = C.T @ C

    norm_sq_err = norm_X_sq

    rank = B.shape[0]
    n_cond = len(X_list)
    n_genes = C.shape[0]

    # Hoist loop-invariant matmuls
    BtB = B.T @ B  # (rank, rank)

    # ---- Pass 1: compute M_i = mat_i @ lhs_i for every condition
    lhs_all = np.einsum("ik,gk,rk->igr", A, C, B, optimize=True)

    M_list = [mat @ lhs_all[i] for i, mat in enumerate(X_list)]

    # ---- Small per-condition linear algebra (rank x rank)
    proj_list = []
    for M in M_list:
        G = M.T @ M  # (rank, rank)
        _, V = np.linalg.eigh(G)
        MV = M @ V  # ≈ U @ S @ D, orthogonal columns
        col_norms = np.linalg.norm(MV, axis=0, keepdims=True)
        safe_norms = np.where(col_norms > 1e-10, col_norms, 1.0)
        proj = (MV / safe_norms) @ V.T  # D cancels → U @ Vh
        proj_list.append(proj)

    if return_projections:
        return proj_list

    # ---- Pass 2: proj_slice_i = proj_i.T @ mat_i
    proj_slice_all = np.stack(
        [p.T @ mat for p, mat in zip(proj_list, X_list, strict=True)],
        axis=0,
    )

    # Allocate the single mttkrp buffer for the requested mode
    if mode == 0:
        mttkrp = np.zeros((n_cond, rank))
    elif mode == 1:
        mttkrp = np.zeros((rank, rank))
    else:
        mttkrp = np.zeros((n_genes, rank))

    for i in range(n_cond):
        proj_slice = proj_slice_all[i]

        B_i_inner = A[i][:, np.newaxis] * BtB * A[i]
        psc = proj_slice @ C  # (rank, rank); needed for error + modes 0,1

        norm_sq_err -= 2.0 * np.einsum("r,jr,jr->", A[i], B, psc)
        norm_sq_err += (B_i_inner * CtC).sum()

        if mode == 0:
            mttkrp[i] = np.sum(psc * B, axis=0)
        elif mode == 1:
            mttkrp += psc * A[i]
        else:
            mttkrp += (proj_slice.T @ B) * A[i]

    return mttkrp, float(norm_sq_err)


def standardize_pf2(
    factors: list[np.ndarray], projections: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    # Order components by condition variance
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
