"""
Low-level numerical routines supporting the PARAFAC2 fit.

Provides norm computation over (optionally mean-centered, optionally sparse)
data, the per-condition projection step, the per-mode ALS factor update
(which forms its own MTTKRP), and post-fit standardization of the factors and
projections.

The fit touches the raw data through exactly two products, which together
dominate runtime on single-cell-sized inputs:

* ``W = (X - 1 mu^T) @ C`` (:func:`calc_W`), which depends only on ``C``.
* ``X^T @ H`` for the mode-2 MTTKRP (inside :func:`parafac_update`).

Everything else flows through the compressed per-condition slices
``S_k = P_k^T W_k``, an ``(n_cond, rank, rank)`` array small enough to keep
resident. In particular the mode-0 and mode-1 MTTKRPs and the reconstruction
error are all functions of ``S`` alone, so the projections and both of those
factor updates can be recomputed from a cached ``W`` without re-reading the
data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse
from tensorly.cp_tensor import cp_flip_sign, cp_normalize

from .backend import matmul, matrix_dtype, rmatmul

if TYPE_CHECKING:
    import anndata
    from scipy.sparse import csr_array


def calc_norm_sq(X: np.ndarray | csr_array, means: np.ndarray | None = None) -> float:
    """Return the squared Frobenius norm of the mean-centered matrix.

    Parameters
    ----------
    X : np.ndarray | csr_array
        The (dense or sparse) matrix to compute the norm of.
    means : np.ndarray | None, default None
        Per-column means to subtract before computing the norm. If ``None``
        or all-zero, ``X`` is used uncentered.

    Returns
    -------
    float
        ``sum((X - means) ** 2)``, computed without densifying a sparse
        ``X``.
    """
    if means is None or np.all(means == 0):
        if issparse(X):
            return float(np.sum(cast("csr_array", X).data ** 2))
        return float(np.sum(X**2))

    means_arr = np.asarray(means).ravel()
    if issparse(X):
        mat_csr = cast("csr_array", X)
        M = mat_csr.shape[0]
        term1 = np.sum(mat_csr.data**2)
        term2 = -2.0 * np.sum(mat_csr.data * means_arr[mat_csr.indices])
        term3 = M * np.sum(means_arr**2)
        return float(term1 + term2 + term3)
    return float(np.sum((X - means_arr) ** 2))


def calc_slice_norms(
    X: np.ndarray | csr_array,
    means: np.ndarray | None,
    condition_unique_idxs: np.ndarray,
    n_cond: int,
) -> np.ndarray:
    """Return the per-condition Frobenius norm of the mean-centered slices.

    Parameters
    ----------
    X : np.ndarray | csr_array
        The (dense or sparse) matrix stacked across all conditions.
    means : np.ndarray | None
        Per-column means to subtract before computing each slice's norm, or
        ``None``/all-zero to skip centering.
    condition_unique_idxs : np.ndarray
        Integer array assigning each row of ``X`` to a condition index in
        ``[0, n_cond)``.
    n_cond : int
        The total number of conditions.

    Returns
    -------
    np.ndarray
        Array of length ``n_cond`` with the Frobenius norm of each
        condition's (mean-centered) rows of ``X``.
    """
    idxs = np.asarray(condition_unique_idxs)
    counts = np.bincount(idxs, minlength=n_cond).astype(np.float64)

    if issparse(X):
        mat_csr = cast("csr_array", X)
        group_of_nnz = np.repeat(idxs, np.diff(mat_csr.indptr))
        sums_sq = np.bincount(
            group_of_nnz, weights=mat_csr.data.astype(np.float64) ** 2, minlength=n_cond
        )
        if means is None or np.all(means == 0):
            return np.sqrt(sums_sq)

        means_arr = np.asarray(means).ravel()
        cross = np.bincount(
            group_of_nnz,
            weights=mat_csr.data.astype(np.float64) * means_arr[mat_csr.indices],
            minlength=n_cond,
        )
        mean_sq_total = np.sum(means_arr**2)
        return np.sqrt(np.maximum(sums_sq - 2.0 * cross + counts * mean_sq_total, 0.0))

    means_arr = np.asarray(means).ravel() if means is not None else 0.0
    row_sums_sq = np.sum((np.asarray(X) - means_arr) ** 2, axis=1)
    return np.sqrt(np.bincount(idxs, weights=row_sums_sq, minlength=n_cond))


def condition_slices(
    condition_unique_idxs: np.ndarray, n_cond: int
) -> list[slice | np.ndarray]:
    """Return a per-condition row selector for each condition.

    Computing ``condition_unique_idxs == i`` inside the per-condition loop
    costs ``O(n_cells)`` per condition, i.e. ``O(n_cells * n_cond)`` per pass
    over the data, plus a fancy-indexed copy each time. Precomputing the
    selectors once drops that to ``O(n_cells)``, and when the rows are
    already grouped by condition (the usual case, since conditions are
    concatenated) the selectors are plain ``slice`` objects, making
    ``W[sel]`` a zero-copy view.

    Parameters
    ----------
    condition_unique_idxs : np.ndarray
        Integer array assigning each row to a condition in ``[0, n_cond)``.
    n_cond : int
        The total number of conditions.

    Returns
    -------
    list[slice | np.ndarray]
        One selector per condition: a ``slice`` when the condition's rows are
        contiguous, otherwise an integer index array.
    """
    idxs = np.asarray(condition_unique_idxs)

    if idxs.size and np.all(np.diff(idxs) >= 0):
        starts = np.searchsorted(idxs, np.arange(n_cond), side="left")
        stops = np.searchsorted(idxs, np.arange(n_cond), side="right")
        return [slice(int(a), int(b)) for a, b in zip(starts, stops, strict=True)]

    order = np.argsort(idxs, kind="stable")
    bounds = np.searchsorted(idxs[order], np.arange(n_cond + 1))
    return [order[bounds[k] : bounds[k + 1]] for k in range(n_cond)]


def calc_W(X: Any, means: np.ndarray | None, C: np.ndarray) -> np.ndarray:
    """Compute ``W = (X - 1 mu^T) @ C``, the first of the two raw-data products.

    ``W`` depends only on ``C``, so it stays valid across the ``A`` and ``B``
    updates and only has to be recomputed once ``C`` changes.

    The product is taken in ``X``'s own dtype. That matters: handing a
    float64 ``C`` to a float32 sparse ``X`` makes SciPy upcast the entire
    sparse matrix, doubling both the memory traffic that dominates this step
    and the peak memory. The result is widened to float64 afterwards, which
    is ``O(n_cells * rank)`` and so negligible beside the product itself.

    Parameters
    ----------
    X : Any
        The (optionally sparse or GPU-backed) data matrix, stacked across all
        conditions, with shape ``(total_cells, n_genes)``.
    means : np.ndarray | None
        Per-gene means to mean-center ``X`` by, or ``None`` to skip centering.
    C : np.ndarray
        The current gene factor matrix, shape ``(n_genes, rank)``.

    Returns
    -------
    np.ndarray
        The float64 array ``W`` of shape ``(total_cells, rank)``.
    """
    C_op = np.ascontiguousarray(C, dtype=matrix_dtype(X))
    W = np.asarray(matmul(X, C_op), dtype=np.float64)
    if means is not None:
        W -= means @ C
    return W


def polar_factor(M: np.ndarray) -> np.ndarray:
    """Compute the nearest orthonormal matrix to M via polar decomposition."""
    G = M.T @ M
    _, V = np.linalg.eigh(G)
    MV = M @ V
    col_norms = np.linalg.norm(MV, axis=0, keepdims=True)
    safe_norms = np.where(col_norms > 1e-10, col_norms, 1.0)
    return (MV / safe_norms) @ V.T


def project_data(
    W: np.ndarray,
    factors: list[np.ndarray],
    cond_slices: list[slice | np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Compute each condition's projection matrix and compressed slice.

    For condition ``k`` the projection ``P_k`` is the orthonormal polar
    factor of ``W_k diag(a_k) B^T``, and the compressed slice is
    ``S_k = P_k^T W_k``. Costs ``O(n_cells * rank^2)`` and touches no raw
    data, so it is roughly two orders of magnitude cheaper than
    :func:`calc_W` and can be repeated freely while ``W`` is cached.

    Parameters
    ----------
    W : np.ndarray
        The cached ``(X - 1 mu^T) @ C`` from :func:`calc_W`.
    factors : list[np.ndarray]
        The current ``[A, B, C]`` factor matrices.
    cond_slices : list[slice | np.ndarray]
        Per-condition row selectors from :func:`condition_slices`.

    Returns
    -------
    tuple[list[np.ndarray], np.ndarray]
        The per-condition projections ``P_k`` (each ``(n_k, rank)`` with
        orthonormal columns), and the stacked compressed slices ``S`` with
        shape ``(n_cond, rank, rank)``.
    """
    A, B = factors[0], factors[1]
    rank = B.shape[0]

    projections: list[np.ndarray] = []
    S = np.empty((len(cond_slices), rank, rank))

    for i, sel in enumerate(cond_slices):
        W_i = W[sel]
        M = W_i @ (B * A[i]).T  # (n_k, rank)
        proj = polar_factor(M)
        projections.append(proj)
        S[i] = proj.T @ W_i

    return projections, S


def calc_err(S: np.ndarray, factors: list[np.ndarray], norm_X_sq: float) -> float:
    """Return the squared reconstruction error from the compressed slices.

    Uses the expansion ``||X||^2 + Tr(A^T A * B^T B * C^T C) - 2 <A, diag(B^T
    S_k)>``, so no raw-data pass is needed and the error is free to evaluate
    as often as desired (e.g. to monitor an inner iteration).

    Parameters
    ----------
    S : np.ndarray
        The stacked compressed slices from :func:`project_data`.
    factors : list[np.ndarray]
        The current ``[A, B, C]`` factor matrices.
    norm_X_sq : float
        The squared Frobenius norm of the mean-centered ``X``, as returned by
        :func:`calc_norm_sq`.

    Returns
    -------
    float
        The squared reconstruction error.
    """
    A, B, C = factors
    norm_sq_err = norm_X_sq + float(((A.T @ A) * (B.T @ B) * (C.T @ C)).sum())
    norm_sq_err -= 2.0 * float(np.sum(A * np.einsum("kqr,qr->kr", S, B)))
    return norm_sq_err


def solve_factors(
    factors: list[np.ndarray],
    mttkrp: np.ndarray,
    mode: int,
) -> list[np.ndarray]:
    """ALS factor update for a single mode using its precomputed MTTKRP."""
    rank = factors[0].shape[1]
    v = np.ones((rank, rank))
    for i, factor in enumerate(factors):
        if i != mode:
            v *= factor.T @ factor

    try:
        factors[mode] = np.linalg.solve(v.T, mttkrp.T).T
    except np.linalg.LinAlgError:
        factors[mode] = np.linalg.lstsq(v.T, mttkrp.T, rcond=None)[0].T

    return factors


def parafac_update(
    factors: list[np.ndarray],
    mode: int,
    S: np.ndarray,
    projections: list[np.ndarray] | None = None,
    *,
    X: Any = None,
    means: np.ndarray | None = None,
    cond_slices: list[slice | np.ndarray] | None = None,
    slice_weights: np.ndarray | None = None,
) -> list[np.ndarray]:
    """
    Form the MTTKRP for the requested mode and update that factor.

    Modes 0 and 1 are built from the compressed slices ``S`` alone and cost
    ``O(n_cond * rank^2)``. Mode 2 is the only update that has to revisit the
    raw data, via ``X^T @ H`` with ``H_k = P_k B diag(a_k)``.

    ``slice_weights``, if given, is a per-condition scalar (e.g. an inverse
    Frobenius norm) applied only to the MTTKRP contributions. This rebalances
    how much each slice contributes to the factor updates without touching or
    copying ``X``, and without affecting the reported error (which
    :func:`calc_err` computes from the unweighted ``S``).

    Parameters
    ----------
    factors : list[np.ndarray]
        The current ``[A, B, C]`` factor matrices; ``factors[mode]`` is
        replaced with the updated matrix.
    mode : int
        Which factor to update (index into ``factors``).
    S : np.ndarray
        The stacked compressed slices from :func:`project_data`.
    projections : list[np.ndarray] | None, default None
        The per-condition projections. Required for ``mode=2`` only.
    X : Any, keyword-only, default None
        The raw data matrix. Required for ``mode=2`` only.
    means : np.ndarray | None, keyword-only, default None
        Per-gene means to mean-center ``X`` by. Used for ``mode=2`` only.
    cond_slices : list[slice | np.ndarray] | None, keyword-only, default None
        Per-condition row selectors from :func:`condition_slices`. Required
        for ``mode=2`` only.
    slice_weights : np.ndarray | None, keyword-only, default None
        Optional per-condition scalar weights, as described above.

    Returns
    -------
    list[np.ndarray]
        ``factors``, with ``factors[mode]`` updated by solving the normal
        equations ``factors[mode] @ v = mttkrp`` for the Gram-matrix product
        ``v`` of the other factors (falling back to a least-squares solve if
        ``v`` is singular).

    Raises
    ------
    ValueError
        If ``mode=2`` is requested without ``projections``, ``X``, or
        ``cond_slices``.
    """
    A, B, _C = factors
    rank = B.shape[0]

    if mode == 0:
        mttkrp = np.einsum("kqr,qr->kr", S, B)
        if slice_weights is not None:
            mttkrp = mttkrp * slice_weights[:, np.newaxis]
    elif mode == 1:
        A_w = A if slice_weights is None else A * slice_weights[:, np.newaxis]
        mttkrp = np.einsum("kqr,kr->qr", S, A_w)
    else:
        if projections is None or X is None or cond_slices is None:
            raise ValueError(
                "mode=2 needs `projections`, `X`, and `cond_slices` to form its MTTKRP."
            )
        # Build H^T directly so the dense operand of the X^T @ H product is
        # C-contiguous and shares X's dtype (see calc_W on why that matters).
        H_T = np.empty((rank, X.shape[0]), dtype=matrix_dtype(X))
        for k, sel in enumerate(cond_slices):
            w_k = 1.0 if slice_weights is None else slice_weights[k]
            H_T[:, sel] = (projections[k] @ (B * A[k]) * w_k).T

        mttkrp_T = np.asarray(rmatmul(H_T, X), dtype=np.float64)
        if means is not None:
            mttkrp_T -= np.outer(H_T.sum(axis=1), means)
        mttkrp = mttkrp_T.T

    return solve_factors(factors, mttkrp, mode)


def standardize_pf2(
    factors: list[np.ndarray], projections: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Put a fitted PARAFAC2 model into a canonical, comparable form.

    Reorders components by condition variance-to-mean ratio, normalizes and
    sign-flips the factors (via TensorLy's ``cp_normalize``/``cp_flip_sign``),
    permutes components to maximize the diagonal of ``B`` (via linear-sum
    assignment), and flips signs so that ``B``'s diagonal is non-negative.

    Parameters
    ----------
    factors : list[np.ndarray]
        The fitted ``[A, B, C]`` factor matrices.
    projections : list[np.ndarray]
        The fitted per-condition projection matrices ``P_k``.

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]
        The ``(weights, factors, projections)`` triple after standardization,
        with components reordered/sign-flipped consistently across
        ``factors`` and ``projections``.
    """
    # Order components by condition variance-to-mean ratio
    mean_a = np.mean(factors[0], axis=0)
    gini = np.divide(
        np.var(factors[0], axis=0),
        mean_a,
        out=np.zeros_like(mean_a),
        where=np.abs(mean_a) > 1e-12,
    )
    gini_idx = np.argsort(gini)
    factors = [f[:, gini_idx] for f in factors]

    weights, factors = cp_flip_sign(cp_normalize((None, factors)), mode=1)

    # Order eigen-cells to maximize the diagonal of B
    _, col_ind = linear_sum_assignment(np.abs(factors[1].T), maximize=True)
    factors[1] = factors[1][col_ind, :]
    projections = [p[:, col_ind] for p in projections]

    # Flip the sign based on B
    signn = np.sign(np.diag(factors[1]))
    factors[1] *= signn[:, np.newaxis]
    projections = [p * signn for p in projections]

    return weights, factors, projections


def randomized_svd_right(
    X: Any,
    means: np.ndarray | None,
    n_components: int,
    n_oversamples: int = 0,
    n_power_iter: int = 2,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Compute the top right-singular vectors of the mean-centered matrix ``(X - 1 mu^T)``.

    Parameters
    ----------
    X : Any
        The (optionally sparse or GPU-backed) data matrix of shape
        ``(total_cells, n_genes)``.
    means : np.ndarray | None
        Per-gene means for implicit centering, or ``None``.
    n_components : int
        Number of right-singular vectors to return.
    n_oversamples : int, default 0
        Additional random test vectors for randomized SVD projection.
    n_power_iter : int, default 2
        Number of power iterations for subspace refinement.
    random_state : int | np.random.Generator | None, default None
        Random seed or NumPy generator.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_genes, n_components)`` with orthonormal columns.
    """
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    n_genes = X.shape[1]
    l_dim = min(n_genes, n_components + n_oversamples)

    Omega = rng.normal(size=(n_genes, l_dim)).astype(np.float64)
    Y = np.asarray(matmul(X, Omega), dtype=np.float64)
    if means is not None:
        Y -= means @ Omega

    for _ in range(n_power_iter):
        Q, _ = np.linalg.qr(Y, mode="reduced")
        Z_T = np.asarray(rmatmul(Q.T, X), dtype=np.float64)
        if means is not None:
            Z_T -= np.outer(np.sum(Q.T, axis=1), means)
        Z = Z_T.T
        Q_z, _ = np.linalg.qr(Z, mode="reduced")
        Y = np.asarray(matmul(X, Q_z), dtype=np.float64)
        if means is not None:
            Y -= means @ Q_z

    Q, _ = np.linalg.qr(Y, mode="reduced")
    B = np.asarray(rmatmul(Q.T, X), dtype=np.float64)
    if means is not None:
        B -= np.outer(np.sum(Q.T, axis=1), means)

    _, _, vh = np.linalg.svd(B, full_matrices=False)
    return vh[:n_components, :].T.astype(np.float64)


def extract_dataset_info(
    X_in: anndata.AnnData,
    normalize_slices: bool = False,
) -> tuple[np.ndarray | csr_array, np.ndarray, np.ndarray, float, np.ndarray | None]:
    """Extract matrix, condition indices, gene means, norm_sq, and optional slice weights.

    Parameters
    ----------
    X_in : anndata.AnnData
        Input single-cell AnnData dataset.
    normalize_slices : bool, default False
        Whether to calculate per-condition slice inverse-norm weights.

    Returns
    -------
    tuple[np.ndarray | csr_array, np.ndarray, np.ndarray, float, np.ndarray | None]
        The ``(X_mat, condition_unique_idxs, means, norm_tensor, slice_weights)`` tuple.
    """
    assert X_in.X is not None
    X_mat = cast("np.ndarray | csr_array", X_in.X)
    condition_unique_idxs = cast(
        "np.ndarray", X_in.obs["condition_unique_idxs"].to_numpy(dtype=int)
    )
    n_cond = int(np.amax(condition_unique_idxs)) + 1

    if "means" in X_in.var:
        means = X_in.var["means"].to_numpy()
    else:
        means = np.zeros(X_mat.shape[1])

    norm_tensor = calc_norm_sq(X_mat, means)

    slice_weights: np.ndarray | None = None
    if normalize_slices:
        slice_norms = calc_slice_norms(X_mat, means, condition_unique_idxs, n_cond)
        slice_weights = np.where(slice_norms > 1e-10, 1.0 / slice_norms, 1.0)

    return X_mat, condition_unique_idxs, means, norm_tensor, slice_weights
