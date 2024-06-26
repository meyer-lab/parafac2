import numpy as np
import cupy as cp
import anndata
import scipy.sparse as sps
from cupyx.scipy import sparse as cupy_sparse
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from scipy.optimize import linear_sum_assignment


def anndata_to_list(X_in: anndata.AnnData) -> list[np.ndarray | sps.csr_array]:
    # Index dataset to a list of conditions
    sgIndex = X_in.obs["condition_unique_idxs"].to_numpy(dtype=int)

    X_list = []
    for i in range(np.amax(sgIndex) + 1):
        # Prepare CuPy matrix
        if isinstance(X_in.X, np.ndarray):
            X_list.append(cp.array(X_in.X[sgIndex == i]))
        else:
            X_list.append(cupy_sparse.csr_matrix(X_in.X[sgIndex == i]))

    return X_list


def project_data(
    X_list: list, means: np.ndarray, factors: list[np.ndarray]
) -> tuple[list[np.ndarray], np.ndarray]:
    A, B, C = factors

    projections: list[np.ndarray] = []
    projected_X = cp.empty((A.shape[0], B.shape[0], C.shape[0]))
    means = cp.array(means)

    for i, mat in enumerate(X_list):
        if isinstance(mat, np.ndarray):
            mat = cp.array(mat)

        lhs = cp.array((A[i] * C) @ B.T, copy=False)
        U, _, Vh = cp.linalg.svd(mat @ lhs - means @ lhs, full_matrices=False)
        proj = U @ Vh

        projections.append(cp.asnumpy(proj))

        # Account for centering
        centering = cp.outer(cp.sum(proj, axis=0), means)
        projected_X[i, :, :] = proj.T @ mat - centering

    return projections, cp.asnumpy(projected_X)


def reconstruction_error(
    factors: list[np.ndarray],
    projections: list[np.ndarray],
    projected_X: np.ndarray,
    norm_X_sq: float,
) -> float:
    """Calculate the reconstruction error from the factors and projected data."""
    A, B, C = factors
    CtC = C.T @ C

    norm_sq_err = norm_X_sq

    for i, proj in enumerate(projections):
        B_i = (proj @ B) * A[i]

        # trace of the multiplication products
        norm_sq_err -= 2.0 * np.trace(A[i][:, np.newaxis] * B.T @ projected_X[i] @ C)
        norm_sq_err += ((B_i.T @ B_i) * CtC).sum()

    return norm_sq_err


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
