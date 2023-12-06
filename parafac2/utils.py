import numpy as np
import cupy as cp
import anndata
import scipy.sparse as sps
from cupyx.scipy import sparse as cupy_sparse
from typing import Sequence
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from scipy.optimize import linear_sum_assignment


def calc_total_norm(X: anndata.AnnData) -> float:
    """Calculate the total norm of the dataset, with centering"""
    Xarr = sps.csr_array(X.X)
    means = X.var["means"].to_numpy()

    # Deal with non-zero values first, by centering
    centered_nonzero = Xarr.data - means[Xarr.indices]
    centered_nonzero_norm = float(np.linalg.norm(centered_nonzero) ** 2.0)

    # Obtain non-zero counts for each column
    # Note that these are sorted, and no column should be empty
    unique, counts = np.unique(Xarr.indices, return_counts=True)
    assert np.all(np.diff(unique) == 1)

    num_zero = Xarr.shape[0] - counts
    assert num_zero.shape == means.shape
    zero_norm = np.sum(np.square(means) * num_zero)

    return zero_norm + centered_nonzero_norm


def project_slices(matrices: Sequence, factors: list):
    A, B, C = factors
    xp = cp.get_array_module(B)

    projections = []
    projected_X = xp.empty((A.shape[0], B.shape[0], C.shape[0]))

    for i, mat in enumerate(matrices):
        mat_gpu = xp.array(mat)

        lhs = B @ (A[i] * C).T
        U, _, Vh = xp.linalg.svd(mat_gpu @ lhs.T, full_matrices=False)
        proj = U @ Vh

        projections.append(proj)
        projected_X[i] = proj.T @ mat_gpu

    return projections, projected_X


def project_data(
    Xarr: sps.csr_array, sgIndex: np.ndarray, means, factors: list
) -> tuple[cp.ndarray, cp.ndarray]:
    A, B, C = factors

    projections = []
    projected_X = cp.empty((A.shape[0], B.shape[0], C.shape[0]))

    for i in range(np.amax(sgIndex) + 1):
        # Prepare CuPy matrix
        mat = cupy_sparse.csr_matrix(Xarr[sgIndex == i], dtype=cp.float32)

        lhs = cp.array(B @ (A[i] * C).T, dtype=cp.float32)
        U, _, Vh = cp.linalg.svd(mat @ lhs.T - means @ lhs.T, full_matrices=False)
        proj = U @ Vh

        projections.append(proj)

        # Account for centering
        centering = cp.outer(cp.sum(proj, axis=0), means)
        projected_X[i, :, :] = proj.T @ mat - centering

    return projections, projected_X


def reconstruction_error(
    factors: list, projections: list, projected_X: cp.ndarray, norm_X_sq: float
):
    """Calculate the reconstruction error from the factors and projected data."""
    A, B, C = factors
    xp = cp.get_array_module(B)
    CtC = C.T @ C

    norm_sq_err = xp.array(norm_X_sq)

    for i, proj in enumerate(projections):
        B_i = (proj @ B) * A[i]

        # trace of the multiplication products
        norm_sq_err -= 2.0 * xp.trace(A[i][:, xp.newaxis] * B.T @ projected_X[i] @ C)
        norm_sq_err += ((B_i.T @ B_i) * CtC).sum()

    return float(cp.asnumpy(norm_sq_err))


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
