from collections.abc import Sequence
from typing import Literal, cast, overload

import anndata
import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cupy_sparse
from scipy.optimize import linear_sum_assignment
from tensorly.cp_tensor import cp_flip_sign, cp_normalize


def parafac_update(
    factors: list[cp.ndarray],
    mttkrps: list[cp.ndarray],
    mode: int,
):
    """
    Perform sequential PARAFAC updates for all modes using pre-computed MTTKRPs.
    This corresponds to Option 2: Sequential with reuse.
    """
    rank = factors[0].shape[1]

    # Compute Gram matrix product using current factors
    v = cp.ones((rank, rank))
    for i, factor in enumerate(factors):
        if i != mode:
            v *= factor.T @ factor

    # Update the factor for the current mode
    factors[mode] = cp.linalg.solve(v.T, mttkrps[mode].T).T

    return factors


def anndata_to_list(X_in: anndata.AnnData) -> list[cp.ndarray | cupy_sparse.csr_matrix]:
    # Index dataset to a list of conditions
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
    means: cp.ndarray,
    factors: list[cp.ndarray],
    norm_X_sq: float,
    return_projections: Literal[True],
) -> list[np.ndarray]: ...


@overload
def project_data(
    X_list: Sequence[cp.ndarray | np.ndarray | cupy_sparse.csr_matrix],
    means: cp.ndarray,
    factors: list[cp.ndarray],
    norm_X_sq: float,
    return_projections: Literal[False] = False,
) -> tuple[list[cp.ndarray], float]: ...


def project_data(
    X_list: Sequence[cp.ndarray | np.ndarray | cupy_sparse.csr_matrix],
    means: cp.ndarray,
    factors: list[cp.ndarray],
    norm_X_sq: float,
    return_projections: bool = False,
) -> list[np.ndarray] | tuple[list[cp.ndarray], float]:
    A, B, C = factors
    CtC = C.T @ C
    assert CtC.dtype == cp.float64

    norm_sq_err = norm_X_sq

    projections: list[np.ndarray] = []
    means = cp.array(means, dtype=cp.float32)

    rank = B.shape[0]
    n_cond = len(X_list)
    n_genes = C.shape[0]
    mttkrps: list[cp.ndarray] = [
        cp.zeros((n_cond, rank), dtype=cp.float32),
        cp.zeros((rank, rank), dtype=cp.float32),
        cp.zeros((n_genes, rank), dtype=cp.float32),
    ]

    for i, mat in enumerate(X_list):
        if isinstance(mat, np.ndarray):
            mat = cp.array(mat, dtype=cp.float32)

        lhs = cp.array((A[i] * C) @ B.T, dtype=cp.float32)
        U, _, Vh = cp.linalg.svd(mat @ lhs - means @ lhs, full_matrices=False)
        proj = U @ Vh
        assert proj.dtype == cp.float32

        if return_projections:
            projections.append(cp.asnumpy(proj))
        else:
            # Account for centering
            centering = cp.outer(cp.sum(proj, axis=0), means)
            proj_slice = proj.T @ mat - centering

            # accumulate error
            B_i_inner = A[i][:, cp.newaxis] * (B.T @ B) * A[i]

            # trace of the multiplication products
            norm_sq_err -= 2.0 * cp.einsum("r,jr,jr->", A[i], B, proj_slice @ C)
            norm_sq_err += (B_i_inner * CtC).sum()

            # Accumulate MTTKRP contributions
            mttkrps[0][i] = cp.sum((proj_slice @ C) * B, axis=0)
            mttkrps[1] += (proj_slice @ C) * A[i]
            mttkrps[2] += (proj_slice.T @ B) * A[i]

    if return_projections:
        return projections

    return mttkrps, float(cp.asnumpy(norm_sq_err))


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
