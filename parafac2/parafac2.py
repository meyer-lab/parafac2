import os
from copy import deepcopy
from typing import Sequence
import numpy as np
import cupy as cp
from tqdm import tqdm
import tensorly as tl
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.tenalg.svd import randomized_svd
from tensorly.decomposition import parafac
from scipy.optimize import linear_sum_assignment


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


def _cmf_reconstruction_error(matrices: Sequence, factors: list, norm_X_sq):
    A, B, C = factors
    xp = cp.get_array_module(B)

    norm_sq_err = norm_X_sq
    CtC = C.T @ C
    projections = []
    projected_X = xp.empty((A.shape[0], B.shape[0], C.shape[0]))

    for i, mat in enumerate(matrices):
        mat_gpu = xp.array(mat)

        lhs = B @ (A[i] * C).T
        U, _, Vh = xp.linalg.svd(mat_gpu @ lhs.T, full_matrices=False)
        proj = U @ Vh

        projections.append(proj)
        projected_X[i] = proj.T @ mat_gpu

        B_i = (proj @ B) * A[i]

        # trace of the multiplication products
        norm_sq_err -= 2.0 * xp.trace(A[i][:, np.newaxis] * B.T @ projected_X[i] @ C)
        norm_sq_err += ((B_i.T @ B_i) * CtC).sum()

    return norm_sq_err, projections, projected_X


def parafac2_init(
    X_in: Sequence,
    rank: int,
    random_state=None,
):
    rng = np.random.RandomState(random_state)

    # Assemble covariance matrix rather than concatenation
    # This saves memory and should be faster
    covM = X_in[0].T @ X_in[0]
    for i in range(1, len(X_in)):
        covM += X_in[i].T @ X_in[i]

    C = randomized_svd(cp.array(covM), rank, random_state=rng, n_iter=4)[0]

    factors = [cp.ones((len(X_in), rank)), cp.eye(rank), C]
    return factors


def parafac2_nd(
    X_in: Sequence,
    rank: int,
    n_iter_max: int = 200,
    tol: float = 1e-6,
    random_state=None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], float]:
    r"""The same interface as regular PARAFAC2."""

    # Verbose if this is not an automated build
    verbose = "CI" not in os.environ

    acc_pow: float = 2.0  # Extrapolate to the iteration^(1/acc_pow) ahead
    acc_fail: int = 0  # How many times acceleration have failed

    norm_tensor = np.sum([np.linalg.norm(xx) ** 2 for xx in X_in])

    tl.set_backend("cupy")
    factors = parafac2_init(X_in, rank, random_state)

    errs = []

    tq = tqdm(range(n_iter_max), disable=(not verbose))
    for iter in tq:
        err, projections, projected_X = _cmf_reconstruction_error(
            X_in, factors, norm_tensor
        )

        # Initiate line search
        if iter % 2 == 0 and iter > 5:
            jump = iter ** (1.0 / acc_pow)

            # Estimate error with line search
            factors_ls = deepcopy(factors)
            factors_ls = [
                factors_old[ii] + (factors[ii] - factors_old[ii]) * jump  # type: ignore
                for ii in range(3)
            ]
            err_ls, projections_ls, projected_X_ls = _cmf_reconstruction_error(
                X_in, factors_ls, norm_tensor
            )

            if err_ls < err:
                acc_fail = 0
                err = err_ls
                projections = projections_ls
                projected_X = projected_X_ls
                factors = factors_ls
            else:
                acc_fail += 1

                if acc_fail >= 4:
                    acc_pow += 1.0
                    acc_fail = 0

                    if verbose:
                        print("Reducing acceleration.")

        errs.append(cp.asnumpy((err / norm_tensor)))

        factors_old = deepcopy(factors)
        _, factors = parafac(
            projected_X,
            rank,
            n_iter_max=3,
            init=(None, factors),  # type: ignore
            tol=False,
            normalize_factors=False,
            l2_reg=0.0001,  # type: ignore
        )

        if iter > 1:
            delta = errs[-2] - errs[-1]
            tq.set_postfix(R2X=1.0 - errs[-1], Δ=delta, refresh=False)

            if delta < tol:
                break

    R2X = 1 - errs[-1]
    tl.set_backend("numpy")

    factors = [cp.asnumpy(f) for f in factors]
    projections = [cp.asnumpy(p) for p in projections]

    return *standardize_pf2(factors, projections), R2X
