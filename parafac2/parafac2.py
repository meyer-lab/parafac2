import os
from collections.abc import Callable
from copy import deepcopy

import anndata
import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cupy_sparse
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, issparse
from tqdm import tqdm

from .utils import (
    anndata_to_list,
    parafac_update,
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
        X.obsm["projections"][sgIndex == i, :] = p

    X.obsm["weighted_projections"] = X.obsm["projections"] @ X.uns["Pf2_B"]

    return X


def parafac2_init(
    X_in: list[np.ndarray | csr_matrix],
    means: np.ndarray,
    rank: int,
    random_state: int | None = None,  # noqa: ARG001
) -> tuple[list[np.ndarray], float]:
    """
    Compute the dataset covariance matrix across all conditions and perform an
    eigendecomposition on CPU to initialize factors.
    """
    # Index dataset to a list of conditions
    n_cond = len(X_in)
    n_genes: int = X_in[0].shape[1]
    means = means.ravel()

    # Calculate covariance matrix while preserving sparsity
    cov_matrix = np.zeros((n_genes, n_genes), dtype=np.float64)
    axis0_sum = np.zeros(n_genes, dtype=np.float64)
    total_rows = 0

    for X_cond in X_in:
        if isinstance(X_cond, cp.ndarray):
            cov_matrix += cp.asnumpy(X_cond.T @ X_cond)
            axis0_sum += cp.asnumpy(X_cond.sum(axis=0)).ravel()
        elif isinstance(X_cond, cupy_sparse.spmatrix):
            cov_matrix += cp.asnumpy((X_cond.T @ X_cond).toarray())
            axis0_sum += cp.asnumpy(X_cond.sum(axis=0)).ravel()
        elif issparse(X_cond):
            cov_matrix += (X_cond.T @ X_cond).toarray()
            axis0_sum += np.asarray(X_cond.sum(axis=0)).ravel()
        else:
            cov_matrix += X_cond.T @ X_cond
            axis0_sum += np.asarray(X_cond.sum(axis=0)).ravel()

        total_rows += X_cond.shape[0]

    cov_matrix -= np.outer(means, axis0_sum)
    cov_matrix -= np.outer(axis0_sum, means)
    cov_matrix += total_rows * np.outer(means, means)

    # Calculate the norm using the covariance matrix
    norm_tensor = np.trace(cov_matrix)

    # Compute the top-`rank` eigenvectors of the covariance matrix
    eigenvals, eigenvecs = eigh(
        cov_matrix, subset_by_index=[n_genes - rank, n_genes - 1]
    )
    # Sort in descending order of eigenvalues
    idx = np.argsort(eigenvals)[::-1]
    eigenvecs = eigenvecs[:, idx]

    # Take the top 'rank' eigenvectors as initial C
    factors = [np.ones((n_cond, rank)), np.eye(rank), eigenvecs[:, :rank]]
    return factors, float(norm_tensor)


def parafac2_nd(
    X_in: anndata.AnnData,
    rank: int,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
    callback: Callable[[int, float, list], None] | None = None,
    l1_c: float = 0.0,
    max_iter_cd: int = 100,
    tol_cd: float = 1e-5,
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
        means = X_in.var["means"].to_numpy()
    else:
        means = np.zeros((1, X_in.shape[1]))

    factors, norm_tensor = parafac2_init(X_list, means, rank, random_state)

    mttkrp, err = project_data(X_list, means, factors, norm_tensor, mode=0)
    errs = [err]

    tq = tqdm(range(n_iter_max), disable=(not verbose), delay=0.5)
    for iteration in tq:
        jump = beta_i + 1.0

        factors_old = deepcopy(factors)

        for mode in range(len(factors)):
            factors = parafac_update(
                factors,
                mttkrp,
                mode,
                l1_c=l1_c,
                max_iter_cd=max_iter_cd,
                tol_cd=tol_cd,
            )
            mttkrp, err = project_data(
                X_list, means, factors, norm_tensor, mode=(mode + 1) % len(factors)
            )

        # Estimate error with line search
        factors_ls = [
            factors_old[ii] + (factors[ii] - factors_old[ii]) * jump for ii in range(3)
        ]
        _, err_ls = project_data(X_list, means, factors_ls, norm_tensor, mode=0)

        if l1_c > 0.0:
            obj = 0.5 * err + l1_c * float(np.sum(np.abs(factors[2])))
            obj_ls = 0.5 * err_ls + l1_c * float(np.sum(np.abs(factors_ls[2])))
            is_better = obj_ls < obj
        else:
            is_better = err_ls < err

        if is_better:
            err = err_ls
            factors = factors_ls

            beta_i = min(beta_i_bar, gamma * beta_i)
            beta_i_bar = max(1.0, gamma_bar * beta_i_bar)
        else:
            beta_i_bar = beta_i
            beta_i = beta_i / eta

        errs.append(err / norm_tensor)

        delta = errs[-2] - errs[-1]
        tq.set_postfix(
            error=errs[-1], R2X=1.0 - errs[-1], Δ=delta, jump=jump, refresh=False
        )
        if callback is not None:
            callback(iteration, errs[-1], factors)

        if 0 <= delta < tol:
            break

    R2X = 1 - errs[-1]
    projections: list[np.ndarray] = project_data(
        X_list, means, factors, norm_tensor, mode=0, return_projections=True
    )

    # Standardize the results and return
    return standardize_pf2(factors, projections), R2X
