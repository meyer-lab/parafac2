"""Property-based tests for the CANDELINC compression routines (``compress``)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from ..compress import compress_cells, compress_genes
from .strategies import condition_groups, dense_matrices, real_arrays

# ---------------------------------------------------------------------------
# compress_genes
# ---------------------------------------------------------------------------


@given(n_genes=st.integers(2, 12), has_means=st.booleans(), data=st.data())
@settings(max_examples=40)
def test_compress_genes_q_is_orthonormal(n_genes, has_means, data):
    """Q's columns must be orthonormal regardless of the target L_g.

    Requires n_cells >= n_genes and a well-conditioned X (a diagonal boost
    over a random matrix, as elsewhere in this suite): a rank-deficient X
    hits the bug in `randomized_svd_right` covered separately by
    `test_randomized_svd_right_returns_the_requested_number_of_columns`.
    """
    n_cells = data.draw(st.integers(n_genes, n_genes + 8))
    L_g = data.draw(st.integers(1, n_genes))

    X = data.draw(real_arrays((n_cells, n_genes), min_value=-3.0, max_value=3.0))
    X = X + np.vstack([np.eye(n_genes), np.zeros((n_cells - n_genes, n_genes))]) * 5.0
    means = data.draw(real_arrays((n_genes,))) if has_means else None

    _X_c, Q, _norm = compress_genes(X, means, L_g=L_g, random_state=0)

    assert Q.shape == (n_genes, L_g)
    np.testing.assert_allclose(Q.T @ Q, np.eye(L_g), atol=1e-6)


@given(n_genes=st.integers(2, 10), data=st.data())
@settings(max_examples=40)
def test_compress_genes_is_lossless_when_target_covers_full_rank(n_genes, data):
    """When L_g >= n_genes, the projection loses no variance: X_c @ Q.T must
    reconstruct the (mean-centered) X exactly, and norm_Xc_sq must equal
    the original squared Frobenius norm.

    Requires n_cells >= n_genes and a well-conditioned X, for the same
    reason as `test_compress_genes_q_is_orthonormal` above.
    """
    n_cells = data.draw(st.integers(n_genes, n_genes + 8))
    X = data.draw(real_arrays((n_cells, n_genes), min_value=-3.0, max_value=3.0))
    X = X + np.vstack([np.eye(n_genes), np.zeros((n_cells - n_genes, n_genes))]) * 5.0
    means = data.draw(real_arrays((n_genes,), min_value=-3.0, max_value=3.0))

    X_c, Q, norm_Xc_sq = compress_genes(X, means, L_g=n_genes, random_state=0)

    reconstructed = X_c @ Q.T
    np.testing.assert_allclose(reconstructed, X - means, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        norm_Xc_sq, np.sum((X - means) ** 2), atol=1e-4, rtol=1e-4
    )


@given(n_cells=st.integers(2, 20), n_genes=st.integers(2, 12), data=st.data())
@settings(max_examples=40)
def test_compress_genes_cannot_increase_variance(n_cells, n_genes, data):
    """Projecting onto a subspace cannot increase the captured squared norm
    beyond the original (mean-centered) matrix's."""
    X = data.draw(real_arrays((n_cells, n_genes)))
    means = data.draw(real_arrays((n_genes,)))
    L_g = data.draw(st.integers(1, n_genes))

    _X_c, _Q, norm_Xc_sq = compress_genes(X, means, L_g=L_g, random_state=0)

    # float32 throughout (not float64) means this can overshoot the exact
    # bound by float32-scale rounding error, not just a fixed epsilon --
    # matches the atol/rtol combination the sibling exactness test above
    # already uses for the same comparison.
    reference = np.sum((X - means) ** 2)
    assert norm_Xc_sq <= reference * (1 + 1e-4) + 1e-4


# ---------------------------------------------------------------------------
# compress_cells
# ---------------------------------------------------------------------------


@given(
    mat=dense_matrices(min_rows=2, max_rows=30, min_cols=1, max_cols=8), data=st.data()
)
@settings(max_examples=40)
def test_compress_cells_qk_orthonormal_and_norm_bounded(mat, data):
    """Whenever a per-condition Q_k is computed, its columns must be
    orthonormal, and the total captured norm can't exceed the raw total."""
    n_rows, _L_g = mat.shape
    idxs = data.draw(condition_groups(n_rows=n_rows, max_cond=4))
    L_c = data.draw(st.integers(1, max(1, n_rows)))

    cores, Q_k_list, norm_cores_sq = compress_cells(mat, idxs, L_c=L_c)

    assert len(cores) == int(np.amax(idxs)) + 1
    assert norm_cores_sq <= np.sum(mat**2) + 1e-6

    for cond_i in range(len(cores)):
        n_k = int(np.sum(idxs == cond_i))
        if Q_k_list is not None and Q_k_list[cond_i] is not None:
            Q_i = Q_k_list[cond_i]
            assert Q_i is not None
            assert Q_i.shape[0] == n_k
            np.testing.assert_allclose(Q_i.T @ Q_i, np.eye(Q_i.shape[1]), atol=1e-6)
            # n_k > L_c is exactly the condition that triggers SVD compression.
            assert n_k > L_c
        else:
            # Uncompressed core: it must be exactly that condition's raw rows.
            np.testing.assert_allclose(cores[cond_i], mat[idxs == cond_i])


@given(
    mat=dense_matrices(min_rows=2, max_rows=20, min_cols=1, max_cols=6), data=st.data()
)
@settings(max_examples=40)
def test_compress_cells_skipped_when_l_c_is_none(mat, data):
    """L_c=None must skip cell compression entirely: cores are raw slices."""
    n_rows, _L_g = mat.shape
    idxs = data.draw(condition_groups(n_rows=n_rows, max_cond=4))

    cores, Q_k_list, norm_cores_sq = compress_cells(mat, idxs, L_c=None)

    assert Q_k_list is None
    for cond_i in range(len(cores)):
        np.testing.assert_allclose(cores[cond_i], mat[idxs == cond_i])
    np.testing.assert_allclose(norm_cores_sq, np.sum(mat**2), atol=1e-6, rtol=1e-6)
