"""
Tests for CANDELINC compression routines (compress module) and integration
with parafac2_nd.
"""

import numpy as np
import pytest
from tensorly.decomposition._parafac2 import _parafac2_reconstruction_error
from tensorly.parafac2_tensor import parafac2_to_slices

from ..compress import compress_dataset
from ..parafac2 import parafac2_nd, store_pf2
from .test_parafac2 import pf2_to_anndata


@pytest.mark.parametrize("compress_mode", ["auto", 15, (20, 15), (20, None)])
def test_parafac2_compressed_exact_recovery(compress_mode):
    """Test that compressed PARAFAC2 recovers noise-free synthetic data."""
    shapes = [(30, 40) for _ in range(5)]
    rank = 3
    rng = np.random.default_rng(42)

    A = rng.uniform(0.5, 1.5, size=(len(shapes), rank))
    B = rng.normal(size=(rank, rank))
    C = rng.normal(size=(shapes[0][1], rank))

    projections = []
    for Ik, _ in shapes:
        P = rng.normal(size=(Ik, rank))
        Q, _ = np.linalg.qr(P)
        projections.append(Q)

    factors = [A, B, C]
    X_slices = parafac2_to_slices((None, factors, projections))
    X_ann = pf2_to_anndata(X_slices, sparse=False)

    (w_fit, f_fit, p_fit), r2x = parafac2_nd(
        X_ann,
        rank=rank,
        random_state=42,
        n_iter_max=150,
        tol=1e-7,
        compress=compress_mode,
    )

    norm_X = np.sum([np.linalg.norm(x) ** 2 for x in X_slices])
    rec_err = _parafac2_reconstruction_error(X_slices, (w_fit, f_fit, p_fit))
    relative_err = rec_err / np.sqrt(norm_X)

    assert r2x > 0.99
    assert relative_err < 0.05

    for P_k in p_fit:
        np.testing.assert_allclose(P_k.T @ P_k, np.eye(rank), atol=1e-5)


def test_parafac2_compressed_rank_sweep():
    """Test compressing a dataset once and fitting multiple ranks."""
    shapes = [(40, 50) for _ in range(4)]
    rng = np.random.default_rng(123)

    X_list = [rng.normal(size=shape) for shape in shapes]
    X_ann = pf2_to_anndata(X_list, sparse=False)

    # Compress once with L_g=25, L_c=25
    compressed = compress_dataset(X_ann, L=25, random_state=123)
    assert compressed.L_g == 25
    assert len(compressed.cores) == 4
    assert compressed.cores[0].shape == (25, 25)

    r2x_prev = 0.0
    for r in [2, 3, 5]:
        (w, f, p), r2x = parafac2_nd(
            compressed, rank=r, random_state=123, n_iter_max=30
        )
        assert len(w) == r
        assert f[0].shape == (4, r)
        assert f[1].shape == (r, r)
        assert f[2].shape == (50, r)
        assert len(p) == 4
        for P_k in p:
            assert P_k.shape == (40, r)
            np.testing.assert_allclose(P_k.T @ P_k, np.eye(r), atol=1e-5)

        assert r2x >= r2x_prev - 0.01  # higher rank explains more variance
        r2x_prev = r2x


def test_parafac2_compressed_sparse_dense_equivalence():
    """Test that sparse and dense AnnData inputs yield identical compressed fits."""
    shapes = [(20, 25) for _ in range(3)]
    rank = 2
    rng = np.random.default_rng(42)

    X_list = []
    for Ik, J in shapes:
        x = rng.normal(size=(Ik, J))
        x[rng.random(x.shape) > 0.6] = 0.0
        X_list.append(x)

    X_dense = pf2_to_anndata(X_list, sparse=False)
    X_sparse = pf2_to_anndata(X_list, sparse=True)

    out_dense, r2x_dense = parafac2_nd(
        X_dense, rank=rank, compress=15, random_state=42, n_iter_max=30
    )
    out_sparse, r2x_sparse = parafac2_nd(
        X_sparse, rank=rank, compress=15, random_state=42, n_iter_max=30
    )

    np.testing.assert_allclose(r2x_dense, r2x_sparse, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_dense[0], out_sparse[0], rtol=1e-4, atol=1e-4)


def test_store_pf2_with_compressed():
    """Test store_pf2 works seamlessly when passed a CompressedData object."""
    shapes = [(15, 20) for _ in range(3)]
    rng = np.random.default_rng(42)
    X_ann = pf2_to_anndata([rng.normal(size=s) for s in shapes], sparse=False)

    compressed = compress_dataset(X_ann, L=10, random_state=42)
    pf2_out, _ = parafac2_nd(compressed, rank=3, random_state=42, n_iter_max=10)

    stored = store_pf2(compressed, pf2_out)
    assert stored is X_ann
    assert "Pf2_weights" in stored.uns
    assert stored.obsm["projections"].shape == (45, 3)
    assert stored.obsm["weighted_projections"].shape == (45, 3)


def test_compressed_normalize_slices():
    """Test normalize_slices=True on compressed fits."""
    shapes = [(60, 20), (10, 20), (10, 20)]
    rng = np.random.default_rng(42)
    X_list = [rng.normal(size=s) for s in shapes]
    X_list[0] *= 6.0
    X_ann = pf2_to_anndata(X_list, sparse=False)

    (_w, _f, p), r2x = parafac2_nd(
        X_ann,
        rank=3,
        compress=15,
        normalize_slices=True,
        random_state=42,
        n_iter_max=30,
    )
    assert 0.0 <= r2x <= 1.0
    for P_k in p:
        np.testing.assert_allclose(P_k.T @ P_k, np.eye(3), atol=1e-5)


def test_compressed_rank_too_large():
    """Test ValueError is raised when rank exceeds compression dimension."""
    shapes = [(20, 20) for _ in range(2)]
    rng = np.random.default_rng(42)
    X_ann = pf2_to_anndata([rng.normal(size=s) for s in shapes], sparse=False)

    with pytest.raises(ValueError, match="cannot exceed gene compression dimension"):
        parafac2_nd(X_ann, rank=10, compress=(5, 5))
