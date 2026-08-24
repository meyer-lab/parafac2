"""
Core PARAFAC2 decomposition routines.

Implements PARAFAC2 initialization (randomized SVD), the alternating-least-
squares fitting loop, and standardization/storage of the fitted factors and
per-condition projections. Operates directly on a single (optionally sparse)
data matrix held in an AnnData object, avoiding per-condition copies.
"""

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
    calc_slice_norms,
    parafac_update,
    project_data,
    standardize_pf2,
)


def store_pf2(
    X: anndata.AnnData,
    parafac2_output: tuple[np.ndarray, list[np.ndarray], list[np.ndarray]],
) -> anndata.AnnData:
    """Store the Pf2 results into the anndata object.

    Parameters
    ----------
    X : anndata.AnnData
        The dataset the factorization was fit on. Must have
        ``X.obs["condition_unique_idxs"]`` set (as produced by
        :func:`~parafac2.normalize.prepare_dataset` or equivalent).
    parafac2_output : tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]
        The ``(weights, factors, projections)`` output of :func:`parafac2_nd`,
        where ``factors`` is the ``[A, B, C]`` factor matrices and
        ``projections`` is the per-condition projection matrices ``P_k``.

    Returns
    -------
    anndata.AnnData
        ``X``, mutated in place, with the weights in ``X.uns["Pf2_weights"]``,
        factors in ``X.uns["Pf2_A"]``/``X.uns["Pf2_B"]``/``X.varm["Pf2_C"]``,
        and per-cell projections in ``X.obsm["projections"]`` and
        ``X.obsm["weighted_projections"]`` (projections composed with ``B``).
    """
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

    Parameters
    ----------
    X : Any
        The (optionally sparse or GPU-backed) data matrix, stacked across
        all conditions, with shape ``(total_cells, n_genes)``.
    condition_unique_idxs : np.ndarray
        Integer array of length ``total_cells`` giving each row's condition
        index.
    rank : int, default 3
        The number of components to compute.
    means : np.ndarray | None, default None
        Per-gene means to mean-center ``X`` by, or ``None`` to skip
        centering.
    random_state : int | np.random.Generator | None, default None
        Seed or generator controlling the random projection used by the
        randomized SVD.
    n_oversamples : int, default 10
        Extra dimensions added to ``rank`` when forming the random
        projection, to improve the accuracy of the randomized SVD.
    n_iter : int, default 2
        Number of power iterations used to refine the random projection.
    norm_tensor : float | None, default None
        Precomputed squared Frobenius norm of the mean-centered ``X``. If
        ``None``, it is computed via :func:`~parafac2.utils.calc_norm_sq`.

    Returns
    -------
    tuple[list[np.ndarray], float]
        The initial ``[A, B, C]`` factor matrices (with ``A`` all-ones,
        ``B`` the identity, and ``C`` the top right-singular vectors of the
        mean-centered ``X``), and the squared Frobenius norm ``norm_tensor``.
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
        """Compute ``(X - 1 mu^T) @ R`` without forming the mean-centered ``X``."""
        Y = (X @ R).astype(np.float64)
        return Y - (means @ R).astype(np.float64) if means is not None else Y

    def centered_rmatmul(L: np.ndarray) -> np.ndarray:
        """Compute ``L @ (X - 1 mu^T)`` without forming the mean-centered ``X``."""
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
    normalize_slices: bool = False,
) -> tuple[tuple[np.ndarray, list[np.ndarray], list[np.ndarray]], float]:
    r"""The same interface as regular PARAFAC2.

    If ``normalize_slices`` is True, each condition's contribution to the
    factor updates is rescaled by the inverse of its (mean-centered)
    Frobenius norm. This prevents conditions with many more cells (or much
    higher variance) from dominating the shared factors, e.g. the ``A``
    matrix. The weighting is computed from small per-condition summary
    statistics and applied to per-condition intermediates only, so ``X`` is
    never copied or modified. The reported error/R2X are unaffected, since
    they are still computed from the unweighted fit.

    Parameters
    ----------
    X_in : anndata.AnnData
        Input dataset with the (optionally sparse) data matrix in ``X_in.X``,
        condition labels in ``X_in.obs["condition_unique_idxs"]``, and
        optionally per-gene means in ``X_in.var["means"]`` (defaults to zero,
        i.e. no centering, if absent).
    rank : int
        The number of components to fit.
    n_iter_max : int, default 100
        Maximum number of ALS iterations.
    tol : float, default 1e-6
        Convergence tolerance: iteration stops once the (non-negative)
        decrease in relative error between successive iterations drops
        below this value.
    random_state : int | None, default None
        Seed controlling the randomized SVD initialization.
    callback : Callable[[int, float, list[np.ndarray]], None] | None, default None
        Optional callback invoked after each iteration with the iteration
        index, the relative error, and the current factor matrices.
    backend : str | None, default None
        Compute backend to run matrix products on: one of ``'mlx'``,
        ``'cupy'``, or ``'cpu'``. If ``None``, the first available
        accelerator is auto-detected (see
        :func:`~parafac2.backend.get_backend`).
    normalize_slices : bool, default False
        Whether to rescale each condition's contribution to the factor
        updates by the inverse of its Frobenius norm, as described above.

    Returns
    -------
    tuple[tuple[np.ndarray, list[np.ndarray], list[np.ndarray]], float]
        A ``((weights, factors, projections), R2X)`` tuple: the standardized
        weights and ``[A, B, C]`` factor matrices, the per-condition
        projection matrices ``P_k``, and the final fraction of variance
        explained.
    """
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

    slice_weights: np.ndarray | None = None
    if normalize_slices:
        n_cond = int(np.amax(sgIndex)) + 1
        slice_norms = calc_slice_norms(X_mat, means, sgIndex, n_cond)
        slice_weights = np.where(slice_norms > 1e-10, 1.0 / slice_norms, 1.0)

    X_raw = to_gpu(X_mat, backend=backend)

    factors, _ = parafac2_init(
        X_raw,
        sgIndex,
        rank=rank,
        means=means,
        random_state=random_state,
        norm_tensor=norm_tensor,
    )

    mttkrp, err = project_data(
        X_raw, sgIndex, means, factors, norm_tensor, mode=0, slice_weights=slice_weights
    )
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
                slice_weights=slice_weights,
            )

        errs.append(err / norm_tensor)

        delta = errs[-2] - errs[-1]
        tq.set_postfix(error=errs[-1], R2X=1.0 - errs[-1], Δ=delta, refresh=False)
        if callback is not None:
            callback(iteration, errs[-1], factors)

        if 0 <= delta < tol:
            break

    R2X = 1 - errs[-1]
    projections: list[np.ndarray] = project_data(
        X_raw,
        sgIndex,
        means,
        factors,
        norm_tensor,
        mode=0,
        return_projections=True,
    )

    # Standardize the results and return
    return standardize_pf2(factors, projections), R2X
