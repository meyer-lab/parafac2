import os
from collections.abc import Callable
from copy import deepcopy

import anndata
import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cupy_sparse
from cupyx.scipy.sparse.linalg import eigsh
from tqdm import tqdm

from .utils import (
    anndata_to_list,
    parafac,
    project_data,
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
    X_in: list[cp.ndarray | cupy_sparse.csr_matrix],
    means: cp.ndarray,
    rank: int,
    random_state: int | None = None,
) -> tuple[list[cp.ndarray], float]:
    # Index dataset to a list of conditions
    n_cond = len(X_in)
    n_genes: int = X_in[0].shape[1]
    means = means.ravel()

    # Initialize the random state for eigsh
    if random_state is not None:
        cp.random.seed(random_state)

    # Calculate covariance matrix while preserving sparsity
    cov_matrix = cp.zeros((n_genes, n_genes), dtype=cp.float64)
    axis0_sum = cp.zeros(n_genes, dtype=cp.float64)
    total_rows = 0

    for X_cond in X_in:
        if isinstance(X_cond, cupy_sparse.csr_matrix):
            XX = X_cond.toarray()
            cov_matrix += XX.T @ XX
        else:
            cov_matrix += X_cond.T @ X_cond

        axis0_sum += X_cond.sum(axis=0).flatten()
        total_rows += X_cond.shape[0]

    cov_matrix -= cp.outer(means, axis0_sum)
    cov_matrix -= cp.outer(axis0_sum, means)
    cov_matrix += total_rows * cp.outer(means, means)

    # Calculate the norm using the covariance matrix
    norm_tensor = cp.trace(cov_matrix)

    # Compute eigenvectors of the covariance matrix
    eigenvals, eigenvecs = eigsh(cov_matrix, k=rank)
    # Sort in descending order of eigenvalues
    idx = cp.argsort(eigenvals)[::-1]
    eigenvecs = eigenvecs[:, idx]

    # Take the top 'rank' eigenvectors as initial C
    factors = [cp.ones((n_cond, rank)), cp.eye(rank), eigenvecs[:, :rank]]
    return factors, float(cp.asnumpy(norm_tensor))


def parafac2_nd(
    X_in: anndata.AnnData,
    rank: int,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
    callback: Callable[[int, float, list], None] | None = None,
) -> tuple[tuple, float]:
    r"""The same interface as regular PARAFAC2."""
    # Verbose if this is not an automated build
    verbose = "CI" not in os.environ

    gamma = 1.1
    gamma_bar = 1.03
    eta = 1.5
    beta_i = 0.05
    beta_i_bar = 1.0

    X_list = anndata_to_list(X_in)

    if "means" in X_in.var:
        means = cp.array(X_in.var["means"].to_numpy())
    else:
        means = cp.zeros((1, X_in.shape[1]))

    factors, norm_tensor = parafac2_init(X_list, means, rank, random_state)

    factors_old = deepcopy(factors)

    projected_X, err = project_data(X_list, means, factors, norm_tensor)
    errs = [err]

    tq = tqdm(range(n_iter_max), disable=(not verbose), delay=0.5)
    for iteration in tq:
        jump = beta_i + 1.0

        # Estimate error with line search
        factors_ls = [
            factors_old[ii] + (factors[ii] - factors_old[ii]) * jump for ii in range(3)
        ]

        projected_X_ls, err_ls = project_data(X_list, means, factors, norm_tensor)

        if err_ls < errs[-1] * norm_tensor:
            err = err_ls
            projected_X = projected_X_ls
            factors = factors_ls

            beta_i = min(beta_i_bar, gamma * beta_i)
            beta_i_bar = max(1.0, gamma_bar * beta_i_bar)
        else:
            beta_i_bar = beta_i
            beta_i = beta_i / eta

            projected_X, err = project_data(X_list, means, factors, norm_tensor)

        errs.append(err / norm_tensor)

        factors_old = deepcopy(factors)
        factors = parafac(
            projected_X,
            factors,
        )

        delta = errs[-2] - errs[-1]
        tq.set_postfix(
            error=errs[-1], R2X=1.0 - errs[-1], Δ=delta, jump=jump, refresh=False
        )
        if callback is not None:
            callback(iteration, errs[-1], factors)

        if delta < tol:
            break

    R2X = 1 - errs[-1]
    projections = project_data(
        X_list, means, factors, norm_tensor, return_projections=True
    )

    # Move back to the CPU
    factors = [cp.asnumpy(f) for f in factors]
    projections = [cp.asnumpy(p) for p in projections]

    # Standardize the results and return
    return standardize_pf2(factors, projections), R2X
