import anndata
import cupy as cp
import numpy as np
import tensorly as tl
from cupyx.scipy import sparse as cupy_sparse
from scipy.optimize import linear_sum_assignment
from tensorly.cp_tensor import cp_flip_sign, cp_normalize


def parafac(
    tensor: cp.ndarray,
    factors: list[cp.ndarray],
    n_iter_max: int = 3,
) -> list[cp.ndarray]:
    """Decomposes a tensor into a set of factor matrices using the PARAFAC algorithm.
    Args:
        tensor (numpy.ndarray): The tensor to decompose.
        factors (list of numpy.ndarray): Initial guess for the factor matrices.
            The length of the list must be equal to the number of modes in the tensor.
            Each element of the list is a numpy.ndarray of shape (size_mode_i, rank),
            where size_mode_i is the size of the tensor along mode i, and rank is
            the desired rank of the decomposition.
        n_iter_max (int, optional): Maximum number of iterations for the algorithm.
            Defaults to 3.
    Returns:
        list of numpy.ndarray: A list containing the factor matrices.
            The order of the factor matrices corresponds to the modes of the input
            tensor. The first factor matrix is scaled by the weights.
    """
    rank = factors[0].shape[1]

    for _ in range(n_iter_max):
        for mode in range(tensor.ndim):
            pinv = cp.ones((rank, rank))
            for i, factor in enumerate(factors):
                if i != mode:
                    pinv *= factor.T @ factor

            # With einsum operations:
            if mode == 0:
                mttkrp = cp.einsum("ijr,jr->ir", tensor @ factors[2], factors[1])
            elif mode == 1:
                mttkrp = cp.einsum("ijr,ir->jr", tensor @ factors[2], factors[0])
            else:  # mode == 2
                mttkrp = cp.einsum("ijk,ir,jr->kr", tensor, factors[0], factors[1])

            factors[mode] = cp.linalg.solve(pinv.T, mttkrp.T).T

    tl.set_backend("cupy")
    weights, factors = cp_normalize((None, factors))
    tl.set_backend("numpy")

    # apply weights to first factor matrix
    factors[0] *= weights[None, :]

    return factors


def anndata_to_list(X_in: anndata.AnnData) -> list[cp.ndarray | cupy_sparse.csr_matrix]:
    # Index dataset to a list of conditions
    sgIndex = X_in.obs["condition_unique_idxs"].to_numpy(dtype=int)

    X_list = []
    for i in range(np.amax(sgIndex) + 1):
        # Prepare CuPy matrix
        if isinstance(X_in.X, np.ndarray):
            X_list.append(cp.array(X_in.X[sgIndex == i], dtype=cp.float32))
        else:
            X_list.append(
                cupy_sparse.csr_matrix(X_in.X[sgIndex == i], dtype=cp.float32)  # type: ignore
            )

    return X_list


def project_data(
    X_list: list[cp.ndarray | np.ndarray],
    means: cp.ndarray,
    factors: list[cp.ndarray],
    norm_X_sq: float,
    return_projections=False,
) -> list[cp.ndarray] | tuple[cp.ndarray, float]:
    A, B, C = factors
    CtC = C.T @ C
    assert CtC.dtype == cp.float64

    norm_sq_err = norm_X_sq

    projections: list[cp.ndarray] = []
    projected_X = cp.empty((A.shape[0], B.shape[0], C.shape[0]), dtype=cp.float32)
    means = cp.array(means, dtype=cp.float32)

    for i, mat in enumerate(X_list):
        if isinstance(mat, np.ndarray):
            mat = cp.array(mat, dtype=cp.float32)

        lhs = cp.array((A[i] * C) @ B.T, dtype=cp.float32)
        U, _, Vh = cp.linalg.svd(mat @ lhs - means @ lhs, full_matrices=False)
        proj = U @ Vh
        assert proj.dtype == cp.float32

        if return_projections:
            projections.append(proj)
        else:
            # Account for centering
            centering = cp.outer(cp.sum(proj, axis=0), means)
            proj_slice = proj.T @ mat - centering

            # accumulate error
            B_i_inner = A[i][:, cp.newaxis] * (B.T @ B) * A[i]

            # trace of the multiplication products
            norm_sq_err -= 2.0 * cp.einsum("r,jr,jr->", A[i], B, proj_slice @ C)
            norm_sq_err += (B_i_inner * CtC).sum()

            # store projection
            projected_X[i] = proj_slice

    if return_projections:
        return projections

    return projected_X, float(cp.asnumpy(norm_sq_err))


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
