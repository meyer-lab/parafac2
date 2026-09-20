"""
Low-level numerical routines supporting the PARAFAC2 fit.

Provides the per-condition projection step, the per-mode ALS factor update
(which forms its own MTTKRP), the randomized SVD used for initialization and
gene compression (SciPy's, driven through a linear operator over the data),
and post-fit standardization of the factors and projections.

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

Nothing here inspects ``X``'s type. ``X`` is any object satisfying the
duck-typed matrix contract in :mod:`parafac2.matrix` -- a NumPy array or
SciPy CSR array wrapped by :func:`~parafac2.matrix.as_matrix`, or a
third-party type (e.g. one of ``vsparse``'s normalized views) implementing
it directly. In particular the matrix carries its own mean-centering, so
``X @ C`` already means ``(X - 1 mu^T) @ C`` and no ``means`` argument is
threaded through these routines.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.linalg import interpolative
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import LinearOperator, lobpcg
from tensorly.cp_tensor import cp_flip_sign, cp_normalize

from .matrix import as_linear_operator, as_matrix

if TYPE_CHECKING:
    import anndata


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


def calc_W(X: Any, C: np.ndarray) -> np.ndarray:
    """Compute ``W = (X - 1 mu^T) @ C``, the first of the two raw-data products.

    ``W`` depends only on ``C``, so it stays valid across the ``A`` and ``B``
    updates and only has to be recomputed once ``C`` changes. Any centering
    and dtype handling belongs to ``X`` itself (see :mod:`parafac2.matrix`).

    Parameters
    ----------
    X : Any
        The data matrix, stacked across all conditions, with shape
        ``(total_cells, n_genes)``.
    C : np.ndarray
        The current gene factor matrix, shape ``(n_genes, rank)``.

    Returns
    -------
    np.ndarray
        The float64 array ``W`` of shape ``(total_cells, rank)``.
    """
    return np.asarray(X @ C, dtype=np.float64)


def polar_factor(M: np.ndarray) -> np.ndarray:
    """Compute the nearest matrix with orthonormal columns to M.

    Uses the thin SVD ``M = U S V^T``, whose orthonormal factor ``U @ V^T``
    is the (Frobenius-norm) nearest matrix with orthonormal columns to M --
    unlike a näive ``M @ V / diag(S)`` reconstruction, this stays exactly
    orthonormal even when M is column-rank-deficient (some singular values
    are zero), since LAPACK's SVD always returns a fully orthonormal ``U``
    regardless of M's rank, filling in arbitrary-but-orthonormal directions
    for the zero-singular-value columns.

    Raises
    ------
    ValueError
        If M has fewer rows than columns: no matrix of that shape can have
        orthonormal columns (there is no room for that many independent
        unit vectors), so this is a genuine shape mismatch upstream (e.g. a
        PARAFAC2 condition with fewer cells than the fit rank) rather than
        something this function can paper over.
    """
    n_rows, rank = M.shape
    if n_rows < rank:
        raise ValueError(
            f"Cannot form an orthonormal {n_rows}x{rank} matrix: orthonormal "
            f"columns require at least as many rows as columns, but only "
            f"{n_rows} are available for {rank} requested. This condition "
            f"has fewer cells than the fit rank; PARAFAC2 requires every "
            f"condition to have at least `rank` cells."
        )
    U, _S, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


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
        # Build H^T directly, in X's own dtype, so the dense operand of the
        # X^T @ H product needs no conversion on its way into the kernel.
        H_T = np.empty((rank, X.shape[0]), dtype=X.dtype)
        for k, sel in enumerate(cond_slices):
            w_k = 1.0 if slice_weights is None else slice_weights[k]
            H_T[:, sel] = (projections[k] @ (B * A[k]) * w_k).T

        mttkrp = np.asarray(H_T @ X, dtype=np.float64).T

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


def _id_right_vectors(
    op: LinearOperator,
    n_components: int,
    n_oversamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Right-singular vectors of ``op`` from an interpolative decomposition.

    Falls back to the leading identity columns when there is nothing to
    decompose: a (numerically) all-zero matrix leaves the decomposition's
    column-norm pivoting dividing by zero, and SciPy then either returns
    non-finite vectors or fails outright on them. Any orthonormal basis spans
    such a matrix's row space equally well, and the LOBPCG refinement starts
    from it just the same.
    """
    k = min(*op.shape, n_components + n_oversamples)
    try:
        V = interpolative.svd(op, k, rng=rng)[2][:, :n_components]
    except (ValueError, np.linalg.LinAlgError):
        V = None

    if V is None or not np.all(np.isfinite(V)):
        return np.eye(op.shape[1], n_components, dtype=np.float64)
    return np.ascontiguousarray(V, dtype=np.float64)


def randomized_svd_right(
    X: Any,
    n_components: int,
    n_oversamples: int = 0,
    n_power_iter: int = 20,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Compute the top right-singular vectors of the mean-centered matrix ``(X - 1 mu^T)``.

    The data is wrapped as a linear operator
    (:func:`~parafac2.matrix.as_linear_operator`) and handed to SciPy:
    :func:`scipy.linalg.interpolative.svd` computes a randomized rank-``k``
    SVD through an interpolative decomposition, and
    :func:`scipy.sparse.linalg.lobpcg` then refines the resulting subspace
    against the Gram operator ``X^T X``, whose top eigenvectors are exactly
    the right-singular vectors being sought. LOBPCG applies the operator to
    the whole block at once, so each refinement iteration costs one pass over
    the data in each direction -- the same budget as a power iteration, but
    with a Rayleigh-Ritz step and a conjugate search direction on top.

    Parameters
    ----------
    X : Any
        The data matrix of shape ``(total_cells, n_genes)``, already
        accounting for any mean-centering (see :mod:`parafac2.matrix`).
    n_components : int
        Number of right-singular vectors to return.
    n_oversamples : int, default 0
        Extra columns to ask the interpolative decomposition for before
        truncating to ``n_components``.
    n_power_iter : int, default 20
        Maximum number of LOBPCG refinement iterations; it stops early once
        the subspace has converged. ``0`` returns the interpolative
        decomposition's vectors unrefined.
    random_state : int | np.random.Generator | None, default None
        Random seed or NumPy generator.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_genes, n_components)`` with orthonormal columns.

    Raises
    ------
    ValueError
        If ``n_components`` exceeds ``min(n_cells, n_genes)``, the maximum
        possible rank of ``X``, and therefore the most orthonormal columns
        its row space can supply.
    """
    n_cells, n_genes = X.shape
    max_components = min(n_cells, n_genes)
    if n_components > max_components:
        raise ValueError(
            f"n_components ({n_components}) cannot exceed the maximum "
            f"possible rank of a {n_cells}x{n_genes} matrix "
            f"({max_components})."
        )

    op = as_linear_operator(X)
    rng = np.random.default_rng(random_state)
    V = _id_right_vectors(op, n_components, n_oversamples, rng)

    # LOBPCG needs a problem comfortably larger than its block size; below
    # that it falls back to densifying the operator, which for a data matrix
    # is exactly what none of this is allowed to do. The subspace is then
    # already most of the row space anyway, so the ID's vectors stand.
    if n_power_iter > 0 and n_genes >= 5 * n_components:
        with warnings.catch_warnings():
            # A fixed iteration budget is the point here, so LOBPCG stopping
            # short of its tolerance is expected rather than noteworthy.
            warnings.simplefilter("ignore", UserWarning)
            _eigenvalues, V = lobpcg(op.H @ op, V, largest=True, maxiter=n_power_iter)

    return np.ascontiguousarray(V, dtype=np.float64)


def extract_dataset_info(
    X_in: anndata.AnnData,
    normalize_slices: bool = False,
) -> tuple[Any, np.ndarray, np.ndarray, float, np.ndarray | None]:
    """Wrap an AnnData's matrix and summarize what the fit needs up front.

    This is the boundary where a dataset stops being a specific storage type
    and becomes an opaque duck-typed matrix: ``X_in.X`` is handed to
    :func:`~parafac2.matrix.as_matrix` together with any per-gene means, and
    everything downstream works through that object alone.

    Parameters
    ----------
    X_in : anndata.AnnData
        Input single-cell AnnData dataset.
    normalize_slices : bool, default False
        Whether to calculate per-condition slice inverse-norm weights.

    Returns
    -------
    tuple[Any, np.ndarray, np.ndarray, float, np.ndarray | None]
        The ``(X, condition_unique_idxs, means, norm_tensor, slice_weights)``
        tuple. ``means`` is returned for bookkeeping only -- it is already
        folded into ``X``.
    """
    assert X_in.X is not None
    condition_unique_idxs = cast(
        "np.ndarray", X_in.obs["condition_unique_idxs"].to_numpy(dtype=int)
    )
    n_cond = int(np.amax(condition_unique_idxs)) + 1

    if "means" in X_in.var:
        means = X_in.var["means"].to_numpy()
    else:
        means = np.zeros(X_in.shape[1])

    X = as_matrix(X_in.X, means)
    norm_tensor = float(X.norm_sq())

    slice_weights: np.ndarray | None = None
    if normalize_slices:
        slice_norms = np.asarray(
            X.slice_norms(condition_unique_idxs, n_cond), dtype=np.float64
        )
        slice_weights = np.where(slice_norms > 1e-10, 1.0 / slice_norms, 1.0)

    return X, condition_unique_idxs, means, norm_tensor, slice_weights
