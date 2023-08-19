import os
from copy import deepcopy
from typing import Sequence
import numpy as np
from tqdm import tqdm
import tensorly as tl
from tensorly.cp_tensor import CPTensor
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.tenalg.svd import randomized_svd
from scipy.linalg import khatri_rao
from scipy.optimize import linear_sum_assignment


def _cmf_reconstruction_error(matrices: Sequence, factors: list, norm_X_sq: float):
    A, B, C = factors

    norm_sq_err = norm_X_sq
    CtC = C.T @ C
    projections = []
    projected_X = []

    for i, mat in enumerate(matrices):
        U, _, V = np.linalg.svd(mat @ (A[i] * C) @ B.T, full_matrices=False)
        proj = U @ V
        projections.append(proj)
        projected_X.append(proj.T @ mat)

        B_i = (proj @ B) * A[i]

        # trace of the multiplication products
        norm_sq_err -= 2.0 * np.trace(A[i][:, np.newaxis] * B.T @ projected_X[-1] @ C)
        norm_sq_err += ((B_i.T @ B_i) * CtC).sum()

    return norm_sq_err, projections, projected_X


def parafac(tensor, rank, init, n_iter_max=20):
    """A simple implementation of ALS."""
    _, factors = init

    unfolded = [tl.unfold(tensor, i) for i in range(np.ndim(tensor))]

    for _ in range(n_iter_max):
        for mode in range(np.ndim(tensor)):
            pinv = np.ones((rank, rank))
            for i, factor in enumerate(factors):
                if i != mode:
                    pinv *= factor.T @ factor

            if mode == 0:
                kr = khatri_rao(factors[1], factors[2])
            elif mode == 1:
                kr = khatri_rao(factors[0], factors[2])
            elif mode == 2:
                kr = khatri_rao(factors[0], factors[1])
            else:
                raise RuntimeError("Should not end up here.")

            mttkrp = unfolded[mode] @ kr
            factors[mode] = np.linalg.solve(pinv.T, mttkrp.T).T

    return CPTensor((None, factors))


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

    tq = tqdm(range(n_iter_max), disable=(not verbose))
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
            init=CP,
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
