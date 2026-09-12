"""Property-based tests for the low-level numerical routines in ``utils``.

These complement the example-based tests in ``test_parafac2.py`` by checking
invariants that should hold across the whole input space (arbitrary shapes,
sparsity patterns, and condition groupings), rather than just the specific
cases picked by hand.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from ..utils import (
    calc_norm_sq,
    calc_slice_norms,
    condition_slices,
    polar_factor,
    randomized_svd_right,
    solve_factors,
    standardize_pf2,
)
from .strategies import (
    condition_groups,
    dense_matrix_with_sparsity,
    real_arrays,
    to_csr,
)

# ---------------------------------------------------------------------------
# calc_norm_sq
# ---------------------------------------------------------------------------


@given(
    mat=dense_matrix_with_sparsity(),
    has_means=st.booleans(),
    data=st.data(),
)
@settings(max_examples=100)
def test_calc_norm_sq_sparse_dense_agree(mat, has_means, data):
    """calc_norm_sq must agree bit-for-bit-close whether X is dense or sparse."""
    means = (
        data.draw(real_arrays((mat.shape[1],)), label="means") if has_means else None
    )

    dense_result = calc_norm_sq(mat, means)
    sparse_result = calc_norm_sq(to_csr(mat), means)

    assert dense_result >= -1e-6  # a squared norm cannot be (meaningfully) negative
    np.testing.assert_allclose(dense_result, sparse_result, rtol=1e-6, atol=1e-6)

    # Cross-check against the direct, un-optimized definition.
    expected = np.sum((mat - (means if means is not None else 0.0)) ** 2)
    np.testing.assert_allclose(dense_result, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# calc_slice_norms
# ---------------------------------------------------------------------------


@given(mat=dense_matrix_with_sparsity(min_rows=2, max_rows=15), data=st.data())
@settings(max_examples=100)
def test_calc_slice_norms_matches_bruteforce(mat, data):
    """calc_slice_norms must match a brute-force per-condition norm, for
    dense and sparse input, with and without centering."""
    n_rows = mat.shape[0]
    idxs = data.draw(condition_groups(n_rows=n_rows), label="idxs")
    n_cond = int(np.amax(idxs)) + 1
    has_means = data.draw(st.booleans(), label="has_means")
    means = (
        data.draw(real_arrays((mat.shape[1],)), label="means") if has_means else None
    )

    expected = np.array(
        [
            np.linalg.norm(mat[idxs == i] - (means if means is not None else 0.0))
            for i in range(n_cond)
        ]
    )

    result_dense = calc_slice_norms(mat, means, idxs, n_cond)
    result_sparse = calc_slice_norms(to_csr(mat), means, idxs, n_cond)

    # The sparse path expands ||X - mu||^2 into sum-of-squares cross terms
    # (to avoid densifying X), which loses precision to cancellation when a
    # slice is close to its own mean -- hence the looser absolute tolerance
    # versus the dense path's direct computation.
    np.testing.assert_allclose(result_dense, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result_sparse, expected, rtol=1e-5, atol=1e-3)


# ---------------------------------------------------------------------------
# condition_slices
# ---------------------------------------------------------------------------


@given(n_rows=st.integers(1, 40), data=st.data())
@settings(max_examples=100)
def test_condition_slices_is_a_partition(n_rows, data):
    """Every row must be selected by exactly one condition's selector, and the
    selected rows must all carry that condition's label -- for both the
    grouped (slice) and arbitrarily-ordered (fancy-index) code paths. Grouped
    input must take the zero-copy `slice` path; shuffled input the
    fancy-index path (an already-sorted permutation of a single condition
    counts as "grouped", so the type is asserted per-selector, not per-run)."""
    idxs = data.draw(condition_groups(n_rows=n_rows), label="grouped")
    shuffle = data.draw(st.booleans(), label="shuffle")
    if shuffle and n_rows > 1:
        perm = data.draw(st.permutations(list(range(n_rows))), label="perm")
        idxs = idxs[perm]

    n_cond = int(np.amax(idxs)) + 1
    is_grouped = bool(np.all(np.diff(idxs) >= 0))
    sels = condition_slices(idxs, n_cond)

    assert len(sels) == n_cond
    expected_type = slice if is_grouped else np.ndarray
    assert all(isinstance(s, expected_type) for s in sels)

    seen = np.zeros(n_rows, dtype=bool)
    for k, sel in enumerate(sels):
        rows = np.arange(n_rows)[sel]
        # No overlap between conditions' row sets.
        assert not seen[rows].any()
        seen[rows] = True
        # Every selected row actually carries label k.
        assert np.all(idxs[rows] == k)

    # Every row was claimed by some condition.
    assert seen.all()


# ---------------------------------------------------------------------------
# polar_factor
# ---------------------------------------------------------------------------


@given(rank=st.integers(min_value=1, max_value=6), data=st.data())
@settings(max_examples=100)
def test_polar_factor_orthonormal_when_full_column_rank(rank, data):
    """When M has full column rank, its polar factor must have orthonormal
    columns (P.T @ P == I): this is the property project_data relies on to
    produce a valid PARAFAC2 projection matrix."""
    n_rows = data.draw(st.integers(rank, rank + 6))  # rows >= rank, full column rank

    # Build a full-column-rank M: a random matrix plus a scaled identity
    # block keeps singular values comfortably away from zero, instead of
    # hoping the random draw happens to be nonsingular. The boost (20) must
    # exceed the draw range's magnitude (5), or a boundary draw of exactly
    # -20 could cancel it out and reintroduce a singular M.
    M = data.draw(real_arrays((n_rows, rank), min_value=-5.0, max_value=5.0))
    M = M + np.vstack([np.eye(rank), np.zeros((n_rows - rank, rank))]) * 20.0

    P = polar_factor(M)
    assert P.shape == M.shape
    np.testing.assert_allclose(P.T @ P, np.eye(rank), atol=1e-6)


def test_polar_factor_orthonormal_even_when_rank_deficient():
    """Regression test for a fixed bug: polar_factor used to silently return
    non-orthonormal columns when M was column-rank-deficient. Its old
    near-zero-norm guard (`col_norms > 1e-10`) substituted 1.0 as the divisor
    instead of normalizing to *any* unit vector, so that column of MV came
    back almost unchanged (near-zero norm) rather than unit norm. This was
    reachable in production: project_data calls polar_factor per-condition,
    and any condition whose cells happened to lie in a lower-dimensional
    subspace produced a rank-deficient M, silently breaking the
    P_k^T @ P_k == I invariant the rest of the algorithm assumes. Fixed by
    computing the polar factor via SVD, whose orthonormal `U` factor is
    exact regardless of M's rank.
    """
    rng = np.random.default_rng(0)
    # 6x3 but rank 2: the third column is a linear combination of the first two.
    M = rng.normal(size=(6, 2))
    M = np.hstack([M, M @ np.array([[1.0], [1.0]])])

    P = polar_factor(M)
    np.testing.assert_allclose(P.T @ P, np.eye(3), atol=1e-6)


def test_polar_factor_raises_when_fewer_rows_than_columns():
    """Orthonormal columns are impossible when M has fewer rows than
    columns (a condition with fewer cells than the fit rank): this must
    raise a clear error rather than returning a bogus result."""
    M = np.zeros((2, 3))
    with pytest.raises(ValueError, match="fewer cells than the fit rank"):
        polar_factor(M)


# ---------------------------------------------------------------------------
# randomized_svd_right
# ---------------------------------------------------------------------------


def test_randomized_svd_right_raises_instead_of_silently_truncating():
    """Regression test for a fixed bug: randomized_svd_right used to
    silently return fewer than n_components columns when n_cells <
    n_components (+ n_oversamples), instead of raising. The random test
    matrix Y = X @ Omega has only n_cells rows, so its column-space rank is
    capped at n_cells no matter how wide Omega is; the QR-based power
    iteration then permanently locked in that narrower subspace, and the
    final `vh[:n_components, :]` slice silently returned whatever narrower
    shape fell out instead of the requested one. Reachable in production:
    compress_genes -> compress_dataset (a condition with fewer cells than
    4*rank, the 'auto' L_g target) or any direct rank/L_g request close to
    the number of samples produced a Q (and therefore a C or compressed
    core) with fewer columns than the caller asked for, silently truncating
    the requested rank instead of erroring. Fixed by validating
    n_components against min(n_cells, n_genes) upfront.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2, 5))  # only 2 rows: max achievable rank is 2

    with pytest.raises(ValueError, match="cannot exceed the maximum possible rank"):
        randomized_svd_right(
            X, None, n_components=4, n_oversamples=0, n_power_iter=2, random_state=0
        )


# ---------------------------------------------------------------------------
# standardize_pf2
# ---------------------------------------------------------------------------


@given(
    n_cond=st.integers(2, 6),
    rank=st.integers(1, 4),
    n_genes=st.integers(1, 8),
    data=st.data(),
)
@settings(max_examples=50)
def test_standardize_pf2_b_diagonal_is_nonnegative(n_cond, rank, n_genes, data):
    """After standardization, B's diagonal must be non-negative by construction."""
    A = data.draw(real_arrays((n_cond, rank), min_value=0.1, max_value=5.0))
    B = data.draw(real_arrays((rank, rank), min_value=-5.0, max_value=5.0))
    C = data.draw(real_arrays((n_genes, rank), min_value=-5.0, max_value=5.0))
    projections = [
        np.linalg.qr(data.draw(real_arrays((5, rank))))[0] for _ in range(n_cond)
    ]

    _weights, factors, _projections = standardize_pf2([A, B, C], projections)

    assert np.all(np.diag(factors[1]) >= -1e-9)


# ---------------------------------------------------------------------------
# solve_factors
# ---------------------------------------------------------------------------

# Excludes a thin band around zero (but not zero itself): a *nonzero* factor
# entry with a magnitude far below this still lets `v = factor.T @ factor`
# underflow towards zero, whose reciprocal then overflows to `inf` in LU
# solves further down -- a floating-point artifact of unrealistically tiny
# magnitudes, not a property of `solve_factors` worth chasing here.
_factor_elements = st.one_of(
    st.just(0.0),
    st.floats(min_value=-3.0, max_value=-1e-2, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-2, max_value=3.0, allow_nan=False, allow_infinity=False),
)


@given(
    n_rows=st.integers(1, 8),
    rank=st.integers(1, 4),
    mode=st.integers(0, 2),
    data=st.data(),
)
@settings(max_examples=100)
def test_solve_factors_recovers_a_known_factor(n_rows, rank, mode, data):
    """Given an mttkrp formed from a *known* target factor and the Gram
    product of the others, solve_factors must recover that target factor --
    whenever that Gram product `v` is well-conditioned enough to invert.

    (`solve_factors` recomputes `v` itself from the *other* factors, so the
    test must build `mttkrp` from that same, un-regularized `v` rather than
    a separately-damped copy, or the two would disagree on a singular `v`.)
    """
    shapes = [(n_rows, rank), (rank, rank), (rank, rank)]
    factors = [
        data.draw(arrays(dtype=np.float64, shape=shapes[i], elements=_factor_elements))
        for i in range(3)
    ]
    target = factors[mode]

    v = np.ones((rank, rank))
    for i, factor in enumerate(factors):
        if i != mode:
            v *= factor.T @ factor
    assume(np.linalg.cond(v) < 1e8)

    mttkrp = target @ v.T

    solved = solve_factors([f.copy() for f in factors], mttkrp, mode)
    np.testing.assert_allclose(solved[mode], target, rtol=1e-4, atol=1e-4)
