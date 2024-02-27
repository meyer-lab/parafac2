import os
from copy import deepcopy
import anndata
import numpy as np
import cupy as cp
from tqdm import tqdm
import tensorly as tl
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import svds
from tensorly.decomposition import parafac
from .utils import (
    reconstruction_error,
    standardize_pf2,
    calc_total_norm,
    project_data,
    anndata_to_list,
)


def parafac2_init(
    X_in: anndata.AnnData,
    rank: int,
    random_state=None,
) -> list[cp.ndarray]:
    # Index dataset to a list of conditions
    sgIndex = X_in.obs["condition_unique_idxs"].to_numpy(dtype=int)
    n_cond = np.amax(sgIndex) + 1

    cp.random.seed(random_state)

    if isinstance(X_in.X, np.ndarray):
        mat = cp.array(X_in.X)
    else:
        mat = csr_matrix(X_in.X)

    _, _, C = svds(mat, rank)

    factors = [cp.ones((n_cond, rank)), cp.eye(rank), C.T]
    return factors


def parafac2_nd(
    X_in: anndata.AnnData,
    rank: int,
    n_iter_max: int = 500,
    tol: float = 1e-9,
    random_state=None,
):
    r"""The same interface as regular PARAFAC2."""
    # Verbose if this is not an automated build
    verbose = "CI" not in os.environ

    gamma = 1.1
    gamma_bar = 1.03
    eta = 1.5
    beta_i = 0.05
    beta_i_bar = 1.0

    norm_tensor = calc_total_norm(X_in)
    factors = parafac2_init(X_in, rank, random_state)
    factors_old = deepcopy(factors)

    X_list = anndata_to_list(X_in)

    if "means" in X_in.var:
        means = cp.array(X_in.var["means"].to_numpy())
    else:
        means = cp.zeros((1, factors[2].shape[0]))

    projections, projected_X = project_data(X_list, means, factors)
    err = reconstruction_error(
        factors, projections, projected_X, norm_tensor
    )
    errs = [err]

    tl.set_backend("cupy")

    tq = tqdm(range(n_iter_max), disable=(not verbose))
    for iter in tq:
        jump = beta_i + 1.0

        # Estimate error with line search
        factors_ls = [
            factors_old[ii] + (factors[ii] - factors_old[ii]) * jump
            for ii in range(3)
        ]

        projections_ls, projected_X_ls = project_data(X_list, means, factors)
        err_ls = reconstruction_error(
            factors_ls, projections_ls, projected_X_ls, norm_tensor
        )

        if err_ls < errs[-1] * norm_tensor:
            err = err_ls
            projections = projections_ls
            projected_X = projected_X_ls
            factors = factors_ls

            beta_i = min(beta_i_bar, gamma * beta_i)
            beta_i_bar = max(1.0, gamma_bar * beta_i_bar)
        else:
            beta_i_bar = beta_i
            beta_i = beta_i / eta

            projections, projected_X = project_data(X_list, means, factors)
            err = reconstruction_error(
                factors, projections, projected_X, norm_tensor
            )

        errs.append(err / norm_tensor)

        factors_old = deepcopy(factors)
        _, factors = parafac(
            projected_X,  # type: ignore
            rank,
            n_iter_max=3,
            init=(None, factors),  # type: ignore
            tol=None,  # type: ignore
            normalize_factors=False,
        )

        delta = errs[-2] - errs[-1]
        tq.set_postfix(R2X=1.0 - errs[-1], Δ=delta, jump=jump, refresh=False)

        if delta < tol:
            break

    R2X = 1 - errs[-1]
    tl.set_backend("numpy")

    factors = [cp.asnumpy(f) for f in factors]
    projections = [cp.asnumpy(p) for p in projections]
    return standardize_pf2(factors, projections), R2X
