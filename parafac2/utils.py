from collections.abc import Sequence
from typing import Literal, cast, overload

import anndata
import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cupy_sparse
from scipy.optimize import linear_sum_assignment
from tensorly.cp_tensor import cp_flip_sign, cp_normalize


def parafac_update(
    factors: list[np.ndarray],
    mttkrp: np.ndarray,
    mode: int,
    l1_c: float = 0.0,
    max_iter_cd: int = 100,
    tol_cd: float = 1e-5,
):
    """
    Perform sequential PARAFAC updates for all modes using pre-computed MTTKRPs.
    This corresponds to Option 2: Sequential with reuse.

    All factors here are rank-sized (n_cond, rank), (rank, rank), or
    (n_genes, rank), so this runs on the CPU with numpy: none of these
    matrices are large enough for GPU dispatch overhead to pay off.
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


def anndata_to_list(X_in: anndata.AnnData) -> list[cp.ndarray | cupy_sparse.csr_matrix]:
    # Index dataset to a list of conditions. These per-condition matrices feed
    # directly into the large data GEMMs in project_data/parafac2_init, which
    # are the actual GPU bottlenecks, so they are staged on the GPU here.
    sgIndex = cast("np.ndarray", X_in.obs["condition_unique_idxs"].to_numpy(dtype=int))

    X_list = []
    for i in range(np.amax(sgIndex) + 1):
        # Prepare CuPy matrix
        if isinstance(X_in.X, np.ndarray):
            X_list.append(cp.array(X_in.X[sgIndex == i], dtype=cp.float32))  # type: ignore
        else:
            X_list.append(
                cupy_sparse.csr_matrix(X_in.X[sgIndex == i], dtype=cp.float32)  # type: ignore
            )

    return X_list


@overload
def project_data(
    X_list: Sequence[cp.ndarray | np.ndarray | cupy_sparse.csr_matrix],
    means: np.ndarray,
    factors: list[np.ndarray],
    norm_X_sq: float,
    mode: int,
    return_projections: Literal[True],
) -> list[np.ndarray]: ...


@overload
def project_data(
    X_list: Sequence[cp.ndarray | np.ndarray | cupy_sparse.csr_matrix],
    means: np.ndarray,
    factors: list[np.ndarray],
    norm_X_sq: float,
    mode: int,
    return_projections: Literal[False] = False,
) -> tuple[np.ndarray, float]: ...


def project_data(
    X_list: Sequence[cp.ndarray | np.ndarray | cupy_sparse.csr_matrix],
    means: np.ndarray,
    factors: list[np.ndarray],
    norm_X_sq: float,
    mode: int,
    return_projections: bool = False,
) -> list[np.ndarray] | tuple[np.ndarray, float]:
    """
    Project each condition's data onto the current factors and accumulate the
    MTTKRP for the requested mode.

    Only the two operations that touch the full per-cell data matrices
    (`mat @ lhs` and `proj.T @ mat`) run on the GPU -- profiling showed these
    are ~10-30x faster on GPU and dominate runtime at realistic data scale.
    Everything else here is bounded by `rank` (or is a single-shot batched
    eigh), so it runs on the CPU with numpy: at rank~20 the fixed overhead of
    dispatching hundreds of tiny per-condition GPU kernels is far larger than
    the actual FLOPs, and numpy is both simpler and faster for these.
    """
    A, B, C = factors
    CtC = C.T @ C
    assert CtC.dtype == np.float64

    norm_sq_err = norm_X_sq

    means = np.asarray(means, dtype=np.float32)
    mean_C = (means @ C).ravel()  # (rank,): precompute once

    rank = B.shape[0]
    n_cond = len(X_list)
    n_genes = C.shape[0]

    # Hoist loop-invariant matmuls
    BtB = B.T @ B  # (rank, rank)

    sizes = [mat.shape[0] for mat in X_list]
    bounds = np.cumsum([0, *sizes])

    # ---- Pass 1: compute M_i = mat_i @ lhs_i - mean_term_i for every
    # condition. lhs/mean_term are rank-sized (CPU), but `mat` is the large
    # per-cell data matrix, so the multiply itself runs on the GPU. Batch the
    # host<->device transfers into a single call each way.
    # (n_cond, n_genes, rank)
    lhs_all = np.einsum("ik,gk,rk->igr", A, C, B, optimize=True)
    mean_term_all = np.einsum("ik,k,rk->ir", A, mean_C, B, optimize=True)

    lhs_all_gpu = cp.asarray(lhs_all)
    mean_term_all_gpu = cp.asarray(mean_term_all)

    M_chunks_gpu = []
    for i, mat in enumerate(X_list):
        if isinstance(mat, np.ndarray):
            mat = cp.array(mat, dtype=cp.float32)
        M_chunks_gpu.append(mat @ lhs_all_gpu[i] - mean_term_all_gpu[i])

    M_all = cp.asnumpy(cp.concatenate(M_chunks_gpu, axis=0)).astype(np.float64)
    M_list = [M_all[bounds[i] : bounds[i + 1]] for i in range(n_cond)]

    # ---- Small per-condition linear algebra (rank x rank), entirely CPU.
    proj_list: list[np.ndarray] = []
    for M_f64 in M_list:
        G = M_f64.T @ M_f64  # (rank, rank) float64
        _, V = np.linalg.eigh(G)
        MV = M_f64 @ V  # ≈ U @ S @ D, orthogonal columns
        col_norms = np.linalg.norm(MV, axis=0, keepdims=True)
        safe_norms = np.where(col_norms > 1e-10, col_norms, 1.0)
        proj = ((MV / safe_norms) @ V.T).astype(np.float32)  # D cancels → U @ Vh
        proj_list.append(proj)

    if return_projections:
        return proj_list

    # ---- Pass 2: proj_slice_i = proj_i.T @ mat_i, again the large-data
    # matmul runs on GPU, batched transfer back to host.
    proj_all_gpu = [cp.asarray(p) for p in proj_list]
    proj_slice_chunks_gpu = []
    for i, mat in enumerate(X_list):
        if isinstance(mat, np.ndarray):
            mat = cp.array(mat, dtype=cp.float32)
        proj_slice_chunks_gpu.append(proj_all_gpu[i].T @ mat)  # (rank, n_genes)

    proj_slice_all = cp.asnumpy(cp.stack(proj_slice_chunks_gpu, axis=0))

    # Allocate the single mttkrp buffer for the requested mode
    if mode == 0:
        mttkrp = np.zeros((n_cond, rank), dtype=np.float32)
    elif mode == 1:
        mttkrp = np.zeros((rank, rank), dtype=np.float32)
    else:
        mttkrp = np.zeros((n_genes, rank), dtype=np.float32)

    for i in range(n_cond):
        proj = proj_list[i]

        # Account for centering
        centering = np.outer(np.sum(proj, axis=0), means)
        proj_slice = proj_slice_all[i] - centering

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
