"""
Core PARAFAC2 decomposition routines.

Implements PARAFAC2 initialization (randomized SVD), the alternating-least-
squares fitting loop, CANDELINC-style compression, and standardization/storage
of the fitted factors and per-condition projections. Operates directly on a
single (optionally sparse) data matrix held in an AnnData object, avoiding
per-condition copies.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import anndata
import numpy as np
from tqdm import tqdm

from .backend import to_gpu
from .compress import (
    CompressedData,
    compress_dataset,
    init_compressed_factors,
    project_data_compressed,
)

if TYPE_CHECKING:
    from scipy.sparse import csr_array

from .utils import (
    calc_err,
    calc_norm_sq,
    calc_slice_norms,
    calc_W,
    condition_slices,
    parafac_update,
    project_data,
    solve_factors,
    standardize_pf2,
)


def store_pf2(
    X: anndata.AnnData | CompressedData,
    parafac2_output: tuple[np.ndarray, list[np.ndarray], list[np.ndarray]],
) -> anndata.AnnData:
    """Store the Pf2 results into the anndata object.

    Parameters
    ----------
    X : anndata.AnnData | CompressedData
        The dataset the factorization was fit on. Must have
        ``X.obs["condition_unique_idxs"]`` set (as produced by
        :func:`~parafac2.normalize.prepare_dataset` or equivalent). If a
        :class:`~parafac2.compress.CompressedData` is provided, factors are
        written to its underlying ``.adata`` object.
    parafac2_output : tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]
        The ``(weights, factors, projections)`` output of :func:`parafac2_nd`,
        where ``factors`` is the ``[A, B, C]`` factor matrices and
        ``projections`` is the per-condition projection matrices ``P_k``.

    Returns
    -------
    anndata.AnnData
        The target AnnData object, mutated in place, with the weights in
        ``X.uns["Pf2_weights"]``, factors in
        ``X.uns["Pf2_A"]``/``X.uns["Pf2_B"]``/``X.varm["Pf2_C"]``, and
        per-cell projections in ``X.obsm["projections"]`` and
        ``X.obsm["weighted_projections"]`` (projections composed with ``B``).
    """
    if isinstance(X, CompressedData):
        if X.adata is None:
            raise ValueError("CompressedData has no associated AnnData object.")
        target_adata = X.adata
        sgIndex = X.condition_unique_idxs
    else:
        target_adata = X
        sgIndex = target_adata.obs["condition_unique_idxs"]

    target_adata.uns["Pf2_weights"] = parafac2_output[0]
    target_adata.uns["Pf2_A"], target_adata.uns["Pf2_B"], target_adata.varm["Pf2_C"] = (
        parafac2_output[1]
    )

    target_adata.obsm["projections"] = np.zeros(
        (target_adata.shape[0], len(target_adata.uns["Pf2_weights"])), dtype=np.float32
    )
    for i, p in enumerate(parafac2_output[2]):
        target_adata.obsm["projections"][sgIndex == i, :] = p

    target_adata.obsm["weighted_projections"] = (
        target_adata.obsm["projections"] @ target_adata.uns["Pf2_B"]
    ).astype(np.float32, copy=False)

    return target_adata


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
        ``None``, it is computed via :func:`~parafac2.utils.calc_norm_sq` achievements.

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


def _fit_parafac2_compressed(
    compressed: CompressedData,
    rank: int,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
    callback: Callable[[int, float, list[np.ndarray]], None] | None = None,
    verbose: bool = True,
) -> tuple[tuple[np.ndarray, list[np.ndarray], list[np.ndarray]], float]:
    """Internal fitting loop over compressed cores."""
    if rank > compressed.L_g:
        raise ValueError(
            f"Rank ({rank}) cannot exceed gene compression dimension L_g ({compressed.L_g})."
        )
    if compressed.Q_k is not None and rank > compressed.max_cell_dim:
        raise ValueError(
            f"Rank ({rank}) cannot exceed max cell compression dimension ({compressed.max_cell_dim})."
        )

    factors = init_compressed_factors(compressed.cores, rank, random_state=random_state)

    mttkrp, err = project_data_compressed(
        compressed.cores,
        factors,
        compressed.norm_tensor,
        mode=0,
        slice_weights=compressed.slice_weights,
    )
    errs = [err]

    tq = tqdm(range(n_iter_max), disable=(not verbose), delay=0.5)
    for iteration in tq:
        for mode in range(len(factors)):
            factors = solve_factors(
                factors,
                mttkrp,
                mode,
            )
            mttkrp, err = project_data_compressed(
                compressed.cores,
                factors,
                compressed.norm_tensor,
                mode=(mode + 1) % len(factors),
                slice_weights=compressed.slice_weights,
            )

        errs.append(err / compressed.norm_tensor)

        delta = errs[-2] - errs[-1]
        tq.set_postfix(error=errs[-1], R2X=1.0 - errs[-1], Δ=delta, refresh=False)
        if callback is not None:
            callback(iteration, errs[-1], factors)

        if 0 <= delta < tol:
            break

    R2X = 1.0 - errs[-1]
    projections_tilde = project_data_compressed(
        compressed.cores,
        factors,
        compressed.norm_tensor,
        mode=0,
        return_projections=True,
    )

    # Reconstruct uncompressed C = Q @ C_L
    A, B, C_L = factors
    C = compressed.Q @ C_L
    full_factors = [A, B, C]

    # Reconstruct projections P_k = Q_k @ P_tilde_k
    if compressed.Q_k is not None:
        projections = [
            (Q_k @ P_tilde) if Q_k is not None else P_tilde
            for Q_k, P_tilde in zip(compressed.Q_k, projections_tilde, strict=True)
        ]
    else:
        projections = projections_tilde

    return standardize_pf2(full_factors, projections), R2X


def parafac2_nd(
    X_in: anndata.AnnData | CompressedData,
    rank: int,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
    callback: Callable[[int, float, list[np.ndarray]], None] | None = None,
    backend: str | None = None,
    normalize_slices: bool = False,
    n_inner: int = 1,
    compress: int | tuple[int, int | None] | str | bool | None = None,
) -> tuple[tuple[np.ndarray, list[np.ndarray], list[np.ndarray]], float]:
    r"""The same interface as regular PARAFAC2 with optional CANDELINC compression.

    If ``compress`` is specified (or if ``X_in`` is already a
    :class:`~parafac2.compress.CompressedData`), PARAFAC2 is fit in the
    compressed subspace (Bro's "compress-then-fit"), eliminating per-sweep raw
    data passes.

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
    X_in : anndata.AnnData | CompressedData
        Input dataset with the (optionally sparse) data matrix in ``X_in.X``,\n        condition labels in ``X_in.obs["condition_unique_idxs"]``, and
        optionally per-gene means in ``X_in.var["means"]`` (defaults to zero,
        i.e. no centering, if absent). Alternatively, a pre-compressed
        :class:`~parafac2.compress.CompressedData` object.
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
    n_inner : int, default 1
        Number of ``(projection, A, B)`` sub-iterations per sweep. These read
        the data only through the cached ``W``, costing ``O(n_cells *
        rank^2)`` against the ``O(nnz * rank)`` of a raw-data product, so
        they are close to free on sparse inputs. Raising ``n_inner`` trades
        that cheap compute for fewer sweeps, and so for fewer raw-data
        passes: on structured test data, ``n_inner=2`` cut the sweeps needed
        to reach a fixed error by ~20%, with little further gain beyond 3.
        Whether that is a net win depends on how strongly the raw-data
        products dominate, so it is worth benchmarking per dataset. The
        default of 1 reproduces the classic one-update-per-mode ALS sweep.
    compress : int | tuple[int, int | None] | str | bool | None, default None
        Compression mode. If ``None`` or ``False`` (default), exact ALS is
        used. If ``"auto"`` or ``True``, sets compression dimensions
        automatically based on ``rank``. If an integer, sets both gene and cell
        compression dimensions to that value. If a tuple ``(L_g, L_c)``, sets
        dimensions separately (pass ``L_c=None`` for gene-only compression).
        Ignored if ``X_in`` is already a
        :class:`~parafac2.compress.CompressedData`.

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

    if isinstance(X_in, CompressedData):
        return _fit_parafac2_compressed(
            X_in,
            rank=rank,
            n_iter_max=n_iter_max,
            tol=tol,
            random_state=random_state,
            callback=callback,
            verbose=verbose,
        )

    if compress is not None and compress is not False:
        compressed = compress_dataset(
            X_in,
            L=compress,
            rank=rank,
            random_state=random_state,
            normalize_slices=normalize_slices,
            backend=backend,
        )
        return _fit_parafac2_compressed(
            compressed,
            rank=rank,
            n_iter_max=n_iter_max,
            tol=tol,
            random_state=random_state,
            callback=callback,
            verbose=verbose,
        )

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

    cond_slices = condition_slices(sgIndex, int(np.amax(sgIndex)) + 1)

    # W depends only on C, so it stays valid across the A and B updates and is
    # recomputed only once C changes. Each sweep therefore costs exactly two
    # raw-data products: this one and the X^T @ H inside the mode-2 update.
    W = calc_W(X_raw, means, factors[2])
    projections, S = project_data(W, factors, cond_slices)
    errs = [calc_err(S, factors, norm_tensor) / norm_tensor]

    tq = tqdm(range(n_iter_max), disable=(not verbose), delay=0.5)
    for iteration in tq:
        # The (P, A, B) block reads the data only through the cached W, so
        # extra inner passes buy convergence at no raw-data cost.
        for _ in range(n_inner):
            factors = parafac_update(factors, 0, S, slice_weights=slice_weights)
            projections, S = project_data(W, factors, cond_slices)
            factors = parafac_update(factors, 1, S, slice_weights=slice_weights)
            projections, S = project_data(W, factors, cond_slices)

        factors = parafac_update(
            factors,
            2,
            S,
            projections,
            X=X_raw,
            means=means,
            cond_slices=cond_slices,
            slice_weights=slice_weights,
        )

        # C changed, so refresh W; this also yields the projections and error
        # for the factors as they stand at the end of this sweep.
        W = calc_W(X_raw, means, factors[2])
        projections, S = project_data(W, factors, cond_slices)
        errs.append(calc_err(S, factors, norm_tensor) / norm_tensor)

        delta = errs[-2] - errs[-1]
        tq.set_postfix(error=errs[-1], R2X=1.0 - errs[-1], Δ=delta, refresh=False)
        if callback is not None:
            callback(iteration, errs[-1], factors)

        if 0 <= delta < tol:
            break

    R2X = 1 - errs[-1]

    # Standardize the results and return
    return standardize_pf2(factors, projections), R2X
