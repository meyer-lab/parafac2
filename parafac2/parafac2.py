import os
from collections.abc import Callable, Sequence
from copy import deepcopy

import anndata
import numpy as np
from tqdm import tqdm

from .sample import SampleArray
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

    X.obsm["projections"] = np.zeros(
        (X.shape[0], len(X.uns["Pf2_weights"])), dtype=np.float32
    )
    for i, p in enumerate(parafac2_output[2]):
        X.obsm["projections"][sgIndex == i, :] = p

    X.obsm["weighted_projections"] = (X.obsm["projections"] @ X.uns["Pf2_B"]).astype(
        np.float32, copy=False
    )

    return X


def parafac2_init(
    X_in: Sequence[SampleArray],
    rank: int = 3,
    random_state: int | np.random.Generator | None = None,
    n_oversamples: int = 10,
    n_iter: int = 2,
) -> tuple[list[np.ndarray], float]:
    """
    Compute initial factors using randomized SVD directly performed on
    the sequence of SampleArray objects without copying full raw data.
    """
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    n_cond = len(X_in)
    n_genes = X_in[0].shape[1]
    sizes = [x.shape[0] for x in X_in]
    bounds = np.cumsum([0, *sizes])

    norm_tensor = float(sum(x.norm_sq() for x in X_in))

    l_dim = min(n_genes, rank + n_oversamples)

    Omega = rng.normal(size=(n_genes, l_dim)).astype(np.float64)
    Y_chunks = [x @ Omega for x in X_in]
    Y = np.concatenate(Y_chunks, axis=0)

    for _ in range(n_iter):
        Q, _ = np.linalg.qr(Y, mode="reduced")
        Z = np.zeros((n_genes, l_dim), dtype=np.float64)
        for i in range(n_cond):
            Q_i = Q[bounds[i] : bounds[i + 1], :]
            Z += (Q_i.T @ X_in[i]).T
        Q_z, _ = np.linalg.qr(Z, mode="reduced")
        Y_chunks = [x @ Q_z for x in X_in]
        Y = np.concatenate(Y_chunks, axis=0)

    Q, _ = np.linalg.qr(Y, mode="reduced")

    B = np.zeros((l_dim, n_genes), dtype=np.float64)
    for i in range(n_cond):
        Q_i = Q[bounds[i] : bounds[i + 1], :]
        B += Q_i.T @ X_in[i]

    _, _, vh = np.linalg.svd(B, full_matrices=False)
    C = vh[:rank, :].T.astype(np.float64)

    factors = [
        np.ones((n_cond, rank), dtype=np.float64),
        np.eye(rank, dtype=np.float64),
        C,
    ]
    return factors, norm_tensor


def parafac2_nd(
    X_in: anndata.AnnData,
    rank: int,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
    callback: Callable[[int, float, list[np.ndarray]], None] | None = None,
    l1_c: float = 0.0,
    max_iter_cd: int = 100,
    tol_cd: float = 1e-5,
) -> tuple[tuple[np.ndarray, list[np.ndarray], list[np.ndarray]], float]:
    r"""The same interface as regular PARAFAC2."""
    # Verbose if this is not an automated build
    verbose = "CI" not in os.environ

    gamma = 1.1
    gamma_bar = 1.03
    eta = 1.5
    beta_i = 0.05
    beta_i_bar = 1.0

    X_list = anndata_to_list(X_in)

    factors, norm_tensor = parafac2_init(X_list, rank, random_state)

    mttkrp, err = project_data(X_list, factors, norm_tensor, mode=0)
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
                X_list, factors, norm_tensor, mode=(mode + 1) % len(factors)
            )

        # Estimate error with line search
        factors_ls = [
            factors_old[ii] + (factors[ii] - factors_old[ii]) * jump for ii in range(3)
        ]
        _, err_ls = project_data(X_list, factors_ls, norm_tensor, mode=0)

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
        X_list, factors, norm_tensor, mode=0, return_projections=True
    )

    # Standardize the results and return
    return standardize_pf2(factors, projections), R2X
