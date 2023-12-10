import os
from copy import deepcopy
import anndata
from sklearn.utils.extmath import randomized_svd
import numpy as np
import cupy as cp
from tqdm import tqdm
import tensorly as tl
from tensorly.decomposition import parafac
from .utils import (
    reconstruction_error,
    standardize_pf2,
    calc_total_norm,
    project_data,
)


def parafac2_init(
    X_in: anndata.AnnData,
    rank: int,
    random_state=None,
) -> list[cp.ndarray]:
    # Index dataset to a list of conditions
    sgIndex = X_in.obs["condition_unique_idxs"].to_numpy(dtype=int)
    n_cond = np.amax(sgIndex) + 1

    _, _, C = randomized_svd(X_in.X, rank, random_state=random_state)  # type: ignore

    factors = [cp.ones((n_cond, rank)), cp.eye(rank), cp.array(C.T)]
    return factors


def parafac2_nd(
    X_in: anndata.AnnData,
    rank: int,
    n_iter_max: int = 200,
    tol: float = 1e-6,
    random_state=None,
):
    r"""The same interface as regular PARAFAC2."""
    # Verbose if this is not an automated build
    verbose = "CI" not in os.environ

    acc_pow: float = 2.0  # Extrapolate to the iteration^(1/acc_pow) ahead
    acc_fail: int = 0  # How many times acceleration have failed

    norm_tensor = calc_total_norm(X_in)
    factors = parafac2_init(X_in, rank, random_state)

    errs: list[float] = []
    projections: list[np.ndarray] = []
    err = float("NaN")

    tl.set_backend("cupy")

    tq = tqdm(range(n_iter_max), disable=(not verbose))
    for iter in tq:
        lineIter = iter % 2 == 0 and iter > 5

        # Initiate line search
        if lineIter:
            jump = iter ** (1.0 / acc_pow)

            # Estimate error with line search
            factors_ls = [
                factors_old[ii] + (factors[ii] - factors_old[ii]) * jump  # type: ignore
                for ii in range(3)
            ]

            projections_ls, projected_X_ls = project_data(X_in, factors)
            err_ls = reconstruction_error(
                factors_ls, projections_ls, projected_X_ls, norm_tensor
            )

            if err_ls < errs[-1] * norm_tensor:
                acc_fail = 0
                err = err_ls
                projections = projections_ls
                projected_X = projected_X_ls
                factors = factors_ls
            else:
                lineIter = False
                acc_fail += 1

                if acc_fail >= 4:
                    acc_pow += 1.0
                    acc_fail = 0

                    if verbose:
                        print("Reducing acceleration.")

        if lineIter is False:
            projections, projected_X = project_data(X_in, factors)
            err = reconstruction_error(factors, projections, projected_X, norm_tensor)

        errs.append(err / norm_tensor)

        factors_old = deepcopy(factors)
        _, factors = parafac(
            projected_X,
            rank,
            n_iter_max=5,
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
    return standardize_pf2(factors, projections), R2X
