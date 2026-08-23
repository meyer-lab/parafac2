import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import anndata
import numpy as np
from tqdm import tqdm

from .backend import to_gpu

if TYPE_CHECKING:
    from scipy.sparse import csr_array

from .utils import (
    calc_norm_sq,
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
    X: Any,
    condition_unique_idxs: np.ndarray,
    rank: int = 3,
    means: np.ndarray | None = None,
    random_state: int | np.random.Generator | None = None,
    n_oversamples: int = 10,
    n_iter: int = 2,
    norm_tensor: float | None = None,
) -> tuple[list[np.ndarray], float]:
    """
    Compute initial factors using randomized SVD directly performed on
    the single input matrix X without copying raw data.
    """
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    n_cond = int(np.amax(condition_unique_idxs)) + 1
    n_genes = X.shape[1]
    if norm_tensor is None:
        norm_tensor = calc_norm_sq(X, means)

    l_dim = min(n_genes, rank + n_oversamples)

    def centered_matmul(R: np.ndarray) -> np.ndarray:
        Y = (X @ R).astype(np.float64)
        return Y - (means @ R).astype(np.float64) if means is not None else Y

    def centered_rmatmul(L: np.ndarray) -> np.ndarray:
        Z_T = (L @ X).astype(np.float64)
        if means is not None:
            Z_T -= np.outer(np.sum(L, axis=1), means).astype(np.float64)
        return Z_T

    Omega = rng.normal(size=(n_genes, l_dim)).astype(np.float64)
    Y = centered_matmul(Omega)

    for _ in range(n_iter):
        Q, _ = np.linalg.qr(Y, mode="reduced")
        Z = centered_rmatmul(Q.T).T
        Q_z, _ = np.linalg.qr(Z, mode="reduced")
        Y = centered_matmul(Q_z)

    Q, _ = np.linalg.qr(Y, mode="reduced")
    B = centered_rmatmul(Q.T)

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
    backend: str | None = None,
) -> tuple[tuple[np.ndarray, list[np.ndarray], list[np.ndarray]], float]:
    r"""The same interface as regular PARAFAC2."""
    # Verbose if this is not an automated build
    verbose = "CI" not in os.environ

    assert X_in.X is not None
    X_mat = cast("np.ndarray | csr_array", X_in.X)
    sgIndex = cast("np.ndarray", X_in.obs["condition_unique_idxs"].to_numpy(dtype=int))

    if "means" in X_in.var:
        means = X_in.var["means"].to_numpy()
    else:
        means = np.zeros(X_in.shape[1])

    norm_tensor = calc_norm_sq(X_mat, means)
    X_raw = to_gpu(X_mat, backend=backend)

    factors, _ = parafac2_init(
        X_raw,
        sgIndex,
        rank=rank,
        means=means,
        random_state=random_state,
        norm_tensor=norm_tensor,
    )

    mttkrp, err = project_data(X_raw, sgIndex, means, factors, norm_tensor, mode=0)
    errs = [err]

    tq = tqdm(range(n_iter_max), disable=(not verbose), delay=0.5)
    for iteration in tq:
        for mode in range(len(factors)):
            factors = parafac_update(
                factors,
                mttkrp,
                mode,
            )
            mttkrp, err = project_data(
                X_raw,
                sgIndex,
                means,
                factors,
                norm_tensor,
                mode=(mode + 1) % len(factors),
            )

        errs.append(err / norm_tensor)

        delta = errs[-2] - errs[-1]
        tq.set_postfix(error=errs[-1], R2X=1.0 - errs[-1], Δ=delta, refresh=False)
        if callback is not None:
            callback(iteration, errs[-1], factors)

        if 0 <= delta < tol:
            break

    R2X = 1 - errs[-1]
    projections: list[np.ndarray] = cast(
        "list[np.ndarray]",
        project_data(
            X_raw,
            sgIndex,
            means,
            factors,
            norm_tensor,
            mode=0,
            return_projections=True,
        ),
    )

    # Standardize the results and return
    return standardize_pf2(factors, projections), R2X
