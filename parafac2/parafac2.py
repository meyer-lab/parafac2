import os
from collections.abc import Callable
from copy import deepcopy

import anndata
import numpy as np
from scipy.sparse.linalg import norm
from sklearn.utils.extmath import randomized_svd
from tensorly.decomposition import constrained_parafac, parafac
from tqdm import tqdm

from .SECSI import SECSI
from .utils import (
    anndata_to_list,
    project_data,
    reconstruction_error,
    standardize_pf2,
)


def store_pf2(
    X: anndata.AnnData,
    parafac2_output: tuple[np.ndarray, list[np.ndarray], list[np.ndarray]],
) -> anndata.AnnData:
    """Store the Pf2 results into the anndata object."""
    sgIndex = X.obs["condition_unique_idxs"]

    X.uns["Pf2_weights"] = parafac2_output[0]
    X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"] = parafac2_output[1]

    X.obsm["projections"] = np.zeros((X.shape[0], len(X.uns["Pf2_weights"])))
    for i, p in enumerate(parafac2_output[2]):
        X.obsm["projections"][sgIndex == i, :] = p  # type: ignore

    X.obsm["weighted_projections"] = X.obsm["projections"] @ X.uns["Pf2_B"]

    return X


def parafac2_init(
    X_in: anndata.AnnData,
    rank: int,
    random_state: int | None = None,
) -> tuple[list[np.ndarray], float]:
    # Index dataset to a list of conditions
    n_cond = len(X_in.obs["condition_unique_idxs"].cat.categories)
    means = X_in.var["means"].to_numpy()

    lmult = X_in.X @ means
    if isinstance(X_in.X, np.ndarray):
        norm_tensor = float(np.linalg.norm(X_in.X) ** 2.0 - 2 * np.sum(lmult))
    else:
        norm_tensor = float(norm(X_in.X) ** 2.0 - 2 * np.sum(lmult))

    _, _, C = randomized_svd(X_in.X, rank, random_state=random_state)  # type: ignore

    factors = [np.ones((n_cond, rank)), np.eye(rank), C.T]
    return factors, norm_tensor


def parafac2_nd(
    X_in: anndata.AnnData,
    rank: int,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    l1=0.0,
    random_state: int | None = None,
    SECSI_solver=False,
    callback: Callable[[int, float, list, list], None] | None = None,
) -> tuple[tuple, float]:
    r"""The same interface as regular PARAFAC2."""
    # Verbose if this is not an automated build
    verbose = "CI" not in os.environ

    gamma = 1.1
    gamma_bar = 1.03
    eta = 1.5
    beta_i = 0.05
    beta_i_bar = 1.0

    factors, norm_tensor = parafac2_init(X_in, rank, random_state)
    factors_old = deepcopy(factors)

    X_list = anndata_to_list(X_in)

    if "means" in X_in.var:
        means = np.array(X_in.var["means"].to_numpy())
    else:
        means = np.zeros((1, factors[2].shape[0]))

    projections, projected_X = project_data(X_list, means, factors)
    err = reconstruction_error(factors, projections, projected_X, norm_tensor)
    errs = [err]

    if SECSI_solver:
        SECSerror, factorOuts = SECSI(projected_X, rank, verbose=False)
        factors = factorOuts[np.argmin(SECSerror)].factors

    print("")
    tq = tqdm(range(n_iter_max), disable=(not verbose), delay=1.0)
    for iteration in tq:
        jump = beta_i + 1.0

        # Estimate error with line search
        factors_ls = [
            factors_old[ii] + (factors[ii] - factors_old[ii]) * jump for ii in range(3)
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
            err = reconstruction_error(factors, projections, projected_X, norm_tensor)

        errs.append(err / norm_tensor)

        factors_old = deepcopy(factors)
        cp_init = (None, factors)

        if l1 > 0.0:
            _, factors = constrained_parafac(
                projected_X,
                rank,
                n_iter_max=20,
                init=cp_init,  # type: ignore
                soft_sparsity={2: l1},
                non_negative={0: True},
            )
        else:
            _, factors = parafac(
                projected_X,
                rank,
                n_iter_max=20,
                init=cp_init,  # type: ignore
                tol=None,  # type: ignore
                normalize_factors=False,
            )

        delta = errs[-2] - errs[-1]
        tq.set_postfix(
            error=errs[-1], R2X=1.0 - errs[-1], Δ=delta, jump=jump, refresh=False
        )
        if callback is not None:
            callback(iteration, errs[-1], factors, projections)

        if delta < tol:
            break

    R2X = 1 - errs[-1]
    return standardize_pf2(factors, projections), R2X
