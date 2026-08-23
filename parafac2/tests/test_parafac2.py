"""
Test the data import.
"""

import anndata
import numpy as np
import pytest
from scipy.sparse import csr_array
from tensorly.decomposition._parafac2 import _parafac2_reconstruction_error
from tensorly.parafac2_tensor import parafac2_to_slices
from tensorly.random import random_parafac2

from ..parafac2 import parafac2_init, parafac2_nd
from ..utils import calc_norm_sq, project_data


def pf2_to_anndata(X_list, sparse=False):
    if sparse:
        X_list = [csr_array(XX) for XX in X_list]

    X_ann = [anndata.AnnData(XX) for XX in X_list]

    X_merged = anndata.concat(
        X_ann,
        label="condition_unique_idxs",
        keys=np.arange(len(X_list)),
        index_unique="-",
    )
    X_merged.var["means"] = np.zeros(X_list[0].shape[1])

    return X_merged


@pytest.mark.parametrize("sparse", [False, True])
def test_init_reprod(sparse: bool):
    """Test for reproducibility with the dense formulation."""
    pf2shape_reprod = [(300, 200)] * 5
    X_reprod: list[np.ndarray] = random_parafac2(pf2shape_reprod, rank=3, full=True)

    X_ann = pf2_to_anndata(X_reprod, sparse=sparse)
    X_mat = X_ann.X
    cond_idxs = X_ann.obs["condition_unique_idxs"].to_numpy(dtype=int)
    means = X_ann.var["means"].to_numpy()

    f1, _ = parafac2_init(X_mat, cond_idxs, rank=3, means=means, random_state=1)
    f2, _ = parafac2_init(X_mat, cond_idxs, rank=3, means=means, random_state=1)

    # assert sizes
    assert f1[0].shape == (len(pf2shape_reprod), 3)
    assert f1[1].shape == (3, 3)
    assert f1[2].shape == (pf2shape_reprod[0][1], 3)

    # Compare both seeds
    for ii in range(3):
        np.testing.assert_allclose(f1[ii], f2[ii], rtol=1e-5, atol=1e-5)

    # Compare both seeds for each mode.
    for mode in range(3):
        m1, _ = project_data(X_mat, cond_idxs, means, f1, 1.0, mode=mode)
        m2, _ = project_data(X_mat, cond_idxs, means, f2, 1.0, mode=mode)
        np.testing.assert_allclose(m1, m2, rtol=1e-5, atol=1e-5)


def test_parafac2_orthonormality():
    """Test that the fitted projection matrices are orthonormal (P_k^T @ P_k = I)."""
    shapes = [(30, 40) for _ in range(5)]
    rank = 3
    rng = np.random.default_rng(42)

    # Generate random data
    X_list = [rng.normal(size=shape) for shape in shapes]
    X_ann = pf2_to_anndata(X_list, sparse=False)

    # Fit PARAFAC2
    (_w, _f, p), _ = parafac2_nd(
        X_ann, rank=rank, random_state=42, n_iter_max=50, tol=1e-6
    )

    # Check orthonormality of projections: P_k^T @ P_k = I
    for P_k in p:
        PtP = P_k.T @ P_k
        np.testing.assert_allclose(PtP, np.eye(rank), atol=1e-5)


def test_parafac2_monotonicity():
    """Test that the reconstruction error decreases monotonically at each iteration."""
    shapes = [(30, 45) for _ in range(4)]
    rank = 3
    rng = np.random.default_rng(12)

    X_list = [rng.normal(size=shape) for shape in shapes]
    X_ann = pf2_to_anndata(X_list, sparse=False)

    errors = []

    def callback(_iteration, error, _factors):
        errors.append(error)

    parafac2_nd(
        X_ann,
        rank=rank,
        random_state=12,
        n_iter_max=50,
        tol=1e-10,
        callback=callback,
    )

    # Check monotonicity
    for i in range(1, len(errors)):
        delta = errors[i - 1] - errors[i]
        # Allow tiny float32 precision noise
        assert delta >= -1e-6, (
            f"Error increased at iteration {i}: {errors[i - 1]} -> {errors[i]} "
            f"(delta={delta})"
        )


def test_parafac2_exact_recovery():
    """Test that the PARAFAC2 model can recover noise-free synthetic data."""
    shapes = [(25, 35) for _ in range(5)]
    rank = 3
    rng = np.random.default_rng(100)

    # Generate known true factors and projections
    A = rng.uniform(0.5, 1.5, size=(len(shapes), rank))
    B = rng.normal(size=(rank, rank))
    C = rng.normal(size=(shapes[0][1], rank))

    projections = []
    for Ik, _ in shapes:
        P = rng.normal(size=(Ik, rank))
        Q, _ = np.linalg.qr(P)
        projections.append(Q)

    factors = [A, B, C]

    # Reconstruct noise-free data
    X_slices = parafac2_to_slices((None, factors, projections))
    X_ann = pf2_to_anndata(X_slices, sparse=False)

    # Fit PARAFAC2
    (w_fit, f_fit, p_fit), r2x = parafac2_nd(
        X_ann, rank=rank, random_state=100, n_iter_max=150, tol=1e-7
    )

    # Verify that the relative reconstruction error from TensorLy is small
    norm_X = np.sum([np.linalg.norm(x) ** 2 for x in X_slices])
    rec_err = _parafac2_reconstruction_error(X_slices, (w_fit, f_fit, p_fit))
    relative_err = rec_err / np.sqrt(norm_X)

    assert r2x > 0.99
    assert relative_err < 0.05


def test_parafac2_sparse_dense_equivalence():
    """Test that sparse and dense data representations yield identical results."""
    shapes = [(15, 20) for _ in range(3)]
    rank = 2
    rng = np.random.default_rng(42)

    # Generate random data with some sparsity (zeros)
    X_list = []
    for Ik, J in shapes:
        x = rng.normal(size=(Ik, J))
        x[rng.random(x.shape) > 0.7] = 0.0  # 30% sparsity
        X_list.append(x)

    X_ann_dense = pf2_to_anndata(X_list, sparse=False)
    X_ann_sparse = pf2_to_anndata(X_list, sparse=True)

    # Fit PARAFAC2 with same random seed
    (w_dense, f_dense, p_dense), r2x_dense = parafac2_nd(
        X_ann_dense, rank=rank, n_iter_max=20, tol=1e-6, random_state=42
    )
    (w_sparse, f_sparse, p_sparse), r2x_sparse = parafac2_nd(
        X_ann_sparse, rank=rank, n_iter_max=20, tol=1e-6, random_state=42
    )

    # Check that weights are identical
    np.testing.assert_allclose(w_dense, w_sparse, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(r2x_dense, r2x_sparse, rtol=1e-5, atol=1e-5)

    # Check factors A, B, C
    for fd, fs in zip(f_dense, f_sparse, strict=True):
        np.testing.assert_allclose(fd, fs, rtol=1e-5, atol=1e-5)

    # Check projections P_k
    for pd, ps in zip(p_dense, p_sparse, strict=True):
        np.testing.assert_allclose(pd, ps, rtol=1e-5, atol=1e-5)


def test_pf2_r2x():
    """Compare R2X values to tensorly implementation"""
    pf2shape = [(50, 200)] * 8
    X: list[np.ndarray] = random_parafac2(pf2shape, rank=3, full=True, random_state=2)
    norm_tensor = float(np.linalg.norm(X) ** 2)

    w, f, _ = random_parafac2(pf2shape, rank=3, random_state=1, normalise_factors=False)

    means = np.zeros(X[0].shape[1])
    X_dense = np.concatenate(X, axis=0)
    cond_idxs = np.concatenate([[i] * s[0] for i, s in enumerate(pf2shape)])

    _, errCMF = project_data(X_dense, cond_idxs, means, f, norm_tensor, mode=0)
    p = project_data(
        X_dense,
        cond_idxs,
        means,
        f,
        norm_tensor,
        mode=0,
        return_projections=True,
    )

    err = _parafac2_reconstruction_error(X, (w, f, p)) ** 2

    np.testing.assert_allclose(err, errCMF, rtol=1e-6, atol=1e-6)


def test_pf2_proj_centering():
    """Test that centering the matrix does not affect the results."""
    shapes = [(25, 300) for _ in range(15)]
    _, factors, projections = random_parafac2(
        shapes=shapes,
        rank=3,
        normalise_factors=False,
        dtype=np.float64,
    )

    X_pf = parafac2_to_slices((None, factors, projections))
    norm_X_sq = float(np.sum(np.array([np.linalg.norm(xx) ** 2.0 for xx in X_pf])))

    means_zero = np.zeros(300)
    X_dense = np.concatenate(X_pf, axis=0)
    cond_idxs = np.concatenate([[i] * s[0] for i, s in enumerate(shapes)])

    projected_X, norm_sq_err = project_data(
        X_dense, cond_idxs, means_zero, factors, norm_X_sq, mode=0
    )

    np.testing.assert_allclose(norm_sq_err / norm_X_sq, 0.0, atol=1e-6)

    # De-mean since we aim to subtract off the means
    means = np.random.randn(X_pf[0].shape[1])
    X_dense_mean = X_dense + means

    projected_X_mean, norm_sq_err_centered = project_data(
        X_dense_mean, cond_idxs, means, factors, norm_X_sq, mode=0
    )

    np.testing.assert_allclose(projected_X, projected_X_mean, rtol=1.0e-4, atol=1.0e-4)
    np.testing.assert_allclose(
        norm_sq_err / norm_X_sq, norm_sq_err_centered / norm_X_sq, atol=1e-6
    )


def test_store_pf2():
    """Test storing PARAFAC2 results into an AnnData object."""
    from ..parafac2 import store_pf2

    shapes = [(10, 15), (12, 15)]
    rank = 3
    rng = np.random.default_rng(42)

    X_list = [rng.normal(size=shape) for shape in shapes]
    X_ann = pf2_to_anndata(X_list, sparse=False)

    (w, f, p), _ = parafac2_nd(X_ann, rank=rank, n_iter_max=5, random_state=42)

    stored_ann = store_pf2(X_ann, (w, f, p))

    assert "Pf2_weights" in stored_ann.uns
    assert "Pf2_A" in stored_ann.uns
    assert "Pf2_B" in stored_ann.uns
    assert "Pf2_C" in stored_ann.varm
    assert "projections" in stored_ann.obsm
    assert "weighted_projections" in stored_ann.obsm

    assert stored_ann.obsm["projections"].shape == (22, rank)
    assert stored_ann.obsm["weighted_projections"].shape == (22, rank)
    assert stored_ann.obsm["projections"].dtype == np.float32
    assert stored_ann.obsm["weighted_projections"].dtype == np.float32


def test_parafac2_no_means():
    """Test parafac2_nd fallback when var['means'] is missing."""
    raw = np.ones((10, 5))
    adata = anndata.AnnData(raw)
    adata.obs["condition_unique_idxs"] = np.array([0] * 5 + [1] * 5)

    (_w, f, _p), _r2x = parafac2_nd(adata, rank=2, n_iter_max=5)
    assert len(f) == 3


def test_calc_norm_sq():
    """Test calc_norm_sq for dense and sparse with and without means."""
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(30, 20)).astype(np.float64)
    raw[rng.random(raw.shape) > 0.3] = 0.0
    sparse_mat = csr_array(raw)

    # Without means
    norm_dense_0 = calc_norm_sq(raw)
    norm_sparse_0 = calc_norm_sq(sparse_mat)
    np.testing.assert_allclose(norm_dense_0, np.sum(raw**2))
    np.testing.assert_allclose(norm_dense_0, norm_sparse_0)

    # With means
    means = rng.normal(size=20).astype(np.float64)
    norm_dense_m = calc_norm_sq(raw, means)
    norm_sparse_m = calc_norm_sq(sparse_mat, means)
    expected_m = np.sum((raw - means) ** 2)
    np.testing.assert_allclose(norm_dense_m, expected_m, rtol=1e-10)
    np.testing.assert_allclose(norm_sparse_m, expected_m, rtol=1e-10)


def test_project_data_sparse_dense_with_means():
    """Test project_data produces identical results for sparse and dense with non-zero means."""
    rng = np.random.default_rng(42)
    shapes = [(20, 25) for _ in range(4)]
    rank = 3

    _w, factors, projections = random_parafac2(shapes, rank=rank, random_state=42)
    X_slices = parafac2_to_slices((None, factors, projections))

    means = rng.normal(size=25)
    X_dense = np.concatenate([x + means for x in X_slices], axis=0)
    X_sparse = csr_array(X_dense)
    cond_idxs = np.concatenate([[i] * s[0] for i, s in enumerate(shapes)])
    norm_sq = calc_norm_sq(X_dense, means)

    for mode in range(3):
        mtt_d, err_d = project_data(
            X_dense, cond_idxs, means, factors, norm_sq, mode=mode
        )
        mtt_s, err_s = project_data(
            X_sparse, cond_idxs, means, factors, norm_sq, mode=mode
        )
        np.testing.assert_allclose(mtt_d, mtt_s, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(err_d, err_s, rtol=1e-5, atol=1e-5)

    proj_d = project_data(
        X_dense, cond_idxs, means, factors, norm_sq, mode=0, return_projections=True
    )
    proj_s = project_data(
        X_sparse, cond_idxs, means, factors, norm_sq, mode=0, return_projections=True
    )
    for pd, ps in zip(proj_d, proj_s):
        np.testing.assert_allclose(pd, ps, rtol=1e-5, atol=1e-5)
