import os
from copy import deepcopy
from typing import Sequence
import torch
import numpy as np
from tqdm import tqdm
import tensorly as tl
from tensorly.cp_tensor import CPTensor
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.tenalg.svd import truncated_svd, randomized_svd
from tensorly.decomposition import parafac
from scipy.optimize import linear_sum_assignment


def _cmf_reconstruction_error(matrices: Sequence, factors: list, norm_X_sq):
    A, B, C = factors

    norm_sq_err = norm_X_sq
    CtC = C.T @ C
    projections = []
    projected_X = []

    for i, mat in enumerate(matrices):
        if isinstance(B, torch.Tensor):
            mat_gpu = torch.tensor(mat).cuda()
        else:
            mat_gpu = mat

        lhs = B @ (A[i] * C).T
        U, _, Vh = truncated_svd(mat_gpu @ lhs.T, A.shape[1])
        proj = U @ Vh

        projections.append(proj)
        projected_X.append(proj.T @ mat_gpu)

        B_i = (proj @ B) * A[i]

        # trace of the multiplication products
        norm_sq_err -= 2.0 * tl.trace(A[i][:, np.newaxis] * B.T @ projected_X[-1] @ C)
        norm_sq_err += ((B_i.T @ B_i) * CtC).sum()

    return norm_sq_err, projections, projected_X


@torch.inference_mode()
def parafac2_nd(
    X_in: Sequence,
    rank: int,
    n_iter_max: int = 200,
    tol: float = 1e-6,
    random_state=None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], float]:
    r"""The same interface as regular PARAFAC2."""
    rng = np.random.RandomState(random_state)

    # Verbose if this is not an automated build
    verbose = "CI" not in os.environ

    acc_pow: float = 2.0  # Extrapolate to the iteration^(1/acc_pow) ahead
    acc_fail: int = 0  # How many times acceleration have failed

    norm_tensor = np.sum([np.linalg.norm(xx) ** 2 for xx in X_in])

    # Checks size of each experiment is bigger than rank
    for i in range(len(X_in)):
        assert np.shape(X_in[i])[0] > rank

    # Checks size of signal measured is bigger than rank
    assert np.shape(X_in[0])[1] > rank

    # Assemble covariance matrix rather than concatenation
    # This saves memory and should be faster
    covM = X_in[0].T @ X_in[0]
    for i in range(1, len(X_in)):
        covM += X_in[i].T @ X_in[i]

    C = randomized_svd(covM, rank, random_state=rng, n_iter=4)[0]

    tl.set_backend("pytorch")
    CP = CPTensor(
        (
            None,
            [
                tl.ones((len(X_in), rank)).cuda().double(),
                tl.eye(rank).cuda().double(),
                torch.tensor(C).cuda().double(),
            ],
        )
    )

    errs = []

    tq = tqdm(range(n_iter_max), disable=(not verbose))
    for iter in tq:
        err, projections, projected_X = _cmf_reconstruction_error(
            X_in, CP.factors, norm_tensor
        )

        # Initiate line search
        if iter % 2 == 0 and iter > 5:
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

                if acc_fail >= 4:
                    acc_pow += 1.0
                    acc_fail = 0

                    if verbose:
                        print("Reducing acceleration.")

        errs.append(tl.to_numpy((err / norm_tensor).cpu()))

        # Project tensor slices
        projected_X = tl.stack(projected_X)

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
    tl.set_backend("numpy")

    gini_idx = giniIndex(tl.to_numpy(CP.factors[0].cpu()))
    assert gini_idx.size == rank

    CP.factors = [f.numpy(force=True)[:, gini_idx] for f in CP.factors]
    CP.weights = CP.weights.numpy(force=True)[gini_idx]

    CP = cp_normalize(cp_flip_sign(CP, mode=1))

    for ii in range(3):
        np.testing.assert_allclose(
            np.linalg.norm(CP.factors[ii], axis=0), 1.0, rtol=1e-2
        )

    # Maximize the diagonal of B
    _, col_ind = linear_sum_assignment(np.abs(CP.factors[1].T), maximize=True)
    CP.factors[1] = CP.factors[1][col_ind, :]
    projections = [p.numpy(force=True)[:, col_ind] for p in projections]

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
