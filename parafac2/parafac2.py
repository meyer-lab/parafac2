import os
from copy import deepcopy
from typing import Sequence
import numpy as np
from tqdm import tqdm
from tensorly.cp_tensor import CPTensor
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.tenalg.svd import randomized_svd, truncated_svd
from tensorly.decomposition import parafac
from scipy.optimize import linear_sum_assignment


def _cmf_reconstruction_error(matrices: Sequence, factors: list, norm_X_sq: float):
    A, B, C = factors

    norm_cmf_sq = 0
    inner_product = 0
    CtC = C.T @ C
    projections = []
    projected_X = []

    for i, mat in enumerate(matrices):
        U, _, Vh = truncated_svd(mat @ (A[i] * C) @ B.T, A.shape[1])
        proj = U @ Vh

        B_i = (proj @ B) * A[i]
        # trace of the multiplication products
        inner_product += np.trace(B_i.T @ mat @ C)
        norm_cmf_sq += ((B_i.T @ B_i) * CtC).sum()
        projections.append(proj)
        projected_X.append(proj.T @ mat)

    return norm_X_sq - 2 * inner_product + norm_cmf_sq, projections, projected_X


def parafac2_nd(
    X_in: Sequence,
    rank: int,
    n_iter_max: int = 200,
    tol: float = 1e-6,
    verbose=None,
    linesearch: bool = True,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], float]:
    r"""The same interface as regular PARAFAC2."""

    # Check if verbose was not set
    if verbose is None:
        # Check if this is an automated build
        verbose = "CI" not in os.environ

    acc_pow: float = 2.0  # Extrapolate to the iteration^(1/acc_pow) ahead
    acc_fail: int = 0  # How many times acceleration have failed
    max_fail: int = 4  # Increase acc_pow with one after max_fail failure

    norm_tensor = float(np.sum([np.linalg.norm(xx) ** 2 for xx in X_in]))

    # Checks size of each experiment is bigger than rank
    for i in range(len(X_in)):
        assert np.shape(X_in[i])[0] > rank

    # Checks size of signal measured is bigger than rank
    assert np.shape(X_in[0])[1] > rank

    # Initialization
    unfolded = np.concatenate(X_in, axis=0).T
    C = randomized_svd(unfolded, rank, random_state=1)[0]

    CP = CPTensor(
        (
            None,
            [
                np.ones((len(X_in), rank)),
                np.eye(rank),
                C,
            ],
        )
    )

    errs = []

    tq = tqdm(range(n_iter_max), disable=(not verbose), mininterval=2)
    projections = None
    for iter in tq:
        err, projections, projected_X = _cmf_reconstruction_error(
            X_in, CP.factors, norm_tensor
        )

        # Will we be performing a line search iteration
        if linesearch and iter % 2 == 0 and iter > 1:
            line_iter = True
        else:
            line_iter = False

        # Initiate line search
        if line_iter:
            jump = iter ** (1.0 / acc_pow)

            # Estimate error with line search
            CP_ls = deepcopy(CP)
            CP_ls.factors = [
                CP_old.factors[ii] + (CP.factors[ii] - CP_old.factors[ii]) * jump
                for ii in range(3)
            ]

            err_ls, projections_ls, projected_X_ls = _cmf_reconstruction_error(
                X_in, CP_ls.factors, norm_tensor
            )

            if err_ls < err:
                acc_fail = 0
                err = err_ls
                projections = projections_ls
                projected_X = projected_X_ls
                CP = CP_ls

            else:
                acc_fail += 1

                if verbose:
                    print(f"iter {iter}: Line search failed for jump of {jump}.")

                if acc_fail == max_fail:
                    acc_pow += 1.0
                    acc_fail = 0

                    if verbose:
                        print(f"iter {iter}: Reducing acceleration.")

        errs.append(err / norm_tensor)

        # Project tensor slices
        projected_X = np.stack(projected_X)

        CP_old: CPTensor = deepcopy(CP)

        CP = parafac(
            projected_X,
            rank,
            n_iter_max=10,
            init=CP,
            tol=False,
            normalize_factors=False,
        )

        if iter > 1:
            delta = errs[-2] - errs[-1]
            tq.set_postfix(R2X=1.0 - errs[-1], Δ=delta, refresh=False)

            if delta < tol:
                break

    R2X = 1 - errs[-1]

    gini_idx = giniIndex(CP.factors[0])
    assert gini_idx.size == rank

    CP.factors = [f[:, gini_idx] for f in CP.factors]
    CP.weights = CP.weights[gini_idx]

    CP = cp_normalize(cp_flip_sign(CP, mode=1))

    for ii in range(3):
        np.testing.assert_allclose(
            np.linalg.norm(CP.factors[ii], axis=0), 1.0, rtol=1e-2
        )

    # Maximize the diagonal of B
    _, col_ind = linear_sum_assignment(np.abs(CP.factors[1].T), maximize=True)
    CP.factors[1] = CP.factors[1][col_ind, :]
    projections = [p[:, col_ind] for p in projections]

    # Flip the sign based on B
    signn = np.sign(np.diag(CP.factors[1]))
    CP.factors[1] *= signn[:, np.newaxis]
    projections = [p * signn for p in projections]

    return CP.weights, CP.factors, projections, R2X


def giniIndex(X: np.ndarray) -> np.ndarray:
    """Calculates the Gini Coeff for each component and returns the index rearrangment"""
    X = np.abs(X)
    gini = np.var(X, axis=0) / np.mean(X, axis=0)

    return np.argsort(gini)
