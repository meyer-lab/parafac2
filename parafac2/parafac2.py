import os
from typing import Optional, Callable
import anndata
import numpy as np
from tqdm import tqdm
from .SECSI import SECSI
from sklearn.utils.extmath import randomized_svd
from .utils import (
    reconstruction_error,
    standardize_pf2,
    calc_total_norm,
    project_data,
    anndata_to_list,
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
    random_state: Optional[int] = None,
) -> list[np.ndarray]:
    # Index dataset to a list of conditions
    sgIndex = X_in.obs["condition_unique_idxs"].to_numpy(dtype=int)
    n_cond = np.amax(sgIndex) + 1

    _, _, C = randomized_svd(X_in.X[0:9000, :], rank, random_state=random_state)  # type: ignore

    factors = [np.ones((n_cond, rank)), np.eye(rank), C.T]
    return factors


def parafac2_nd(
    X_in: anndata.AnnData,
    rank: int,
    n_iter_max: int = 25,
    tol: float = 1e-6,
    random_state: Optional[int] = None,
    SECSI_solver=False,
    callback: Optional[Callable[[int, float, list, list], None]] = None,
) -> tuple[tuple, float]:
    r"""The same interface as regular PARAFAC2."""
    # Verbose if this is not an automated build
    verbose = "CI" not in os.environ

    norm_tensor = calc_total_norm(X_in)
    factors = parafac2_init(X_in, rank, random_state)

    X_list = anndata_to_list(X_in)

    if "means" in X_in.var:
        means = np.array(X_in.var["means"].to_numpy())
    else:
        means = np.zeros((1, factors[2].shape[0]))

    projections, projected_X = project_data(X_list, means, factors)
    err = reconstruction_error(factors, projections, projected_X, norm_tensor)
    errs = [err]

    tq = tqdm(range(n_iter_max), disable=(not verbose))
    for iteration in tq:
        projections, projected_X = project_data(X_list, means, factors)
        err = reconstruction_error(factors, projections, projected_X, norm_tensor)

        errs.append(err / norm_tensor)

        SECSerror, factorOuts = SECSI(projected_X, rank, verbose=False)
        factors = factorOuts[np.argmin(SECSerror)].factors

        delta = errs[-2] - errs[-1]
        tq.set_postfix(error=errs[-1], R2X=1.0 - errs[-1], Δ=delta, refresh=False)
        if callback is not None:
            callback(iteration, errs[-1], factors, projections)

        if delta < tol:
            break

    R2X = 1 - errs[-1]
    return standardize_pf2(factors, projections), R2X
