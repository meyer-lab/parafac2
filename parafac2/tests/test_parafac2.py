"""
Test the data import.
"""

import anndata
import cupy as cp
import numpy as np
import pytest
from scipy.sparse import csr_array
from tensorly.decomposition._parafac2 import _parafac2_reconstruction_error
from tensorly.parafac2_tensor import parafac2_to_slices
from tensorly.random import random_parafac2

from ..parafac2 import parafac2_init, parafac2_nd
from ..utils import project_data


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
    means = cp.array(X_ann.var["means"])

    X_list = [cp.array(x) for x in X_reprod]

    f1, _ = parafac2_init(X_list, means, rank=3, random_state=1)
    f2, _ = parafac2_init(X_list, means, rank=3, random_state=1)

    # assert sizes
    assert f1[0].shape == (len(pf2shape_reprod), 3)
    assert f1[1].shape == (3, 3)
    assert f1[2].shape == (pf2shape_reprod[0][1], 3)

    # Compare both seeds
    for ii in range(3):
        cp.testing.assert_array_equal(f1[ii], f2[ii])

    mttkrp_1, _ = project_data(X_reprod, means, f1, 1.0)
    mttkrp_2, _ = project_data(X_reprod, means, f2, 1.0)

    # Compare both seeds
    cp.testing.assert_array_equal(mttkrp_1[0], mttkrp_2[0])
    cp.testing.assert_array_equal(mttkrp_1[1], mttkrp_2[1])
    cp.testing.assert_array_equal(mttkrp_1[2], mttkrp_2[2])


def test_parafac2_orthonormality():
    """Test that the fitted projection matrices are orthonormal (P_k^T @ P_k = I)."""
    shapes = [(30, 40) for _ in range(5)]
    rank = 3
    rng = np.random.default_rng(42)

    # Generate random data
    X_list = [rng.normal(size=shape) for shape in shapes]
    X_ann = pf2_to_anndata(X_list, sparse=False)

    # Fit PARAFAC2
    (w, f, p), _ = parafac2_nd(
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
    cp_f = [cp.array(x) for x in f]

    _, errCMF = project_data(X, cp.zeros((1, X[0].shape[1])), cp_f, norm_tensor)
    p = project_data(
        X, cp.zeros((1, X[0].shape[1])), cp_f, norm_tensor, return_projections=True
    )

    p = [cp.asnumpy(pp) for pp in p]

    err = _parafac2_reconstruction_error(X, (w, f, p)) ** 2

    np.testing.assert_allclose(err, errCMF, rtol=1e-6, atol=1e-6)


def test_pf2_proj_centering():
    """Test that centering the matrix does not affect the results."""
    _, factors, projections = random_parafac2(
        shapes=[(25, 300) for _ in range(15)],
        rank=3,
        normalise_factors=False,
        dtype=np.float64,
    )
    cp_factors = [cp.array(x) for x in factors]

    X_pf = parafac2_to_slices((None, factors, projections))

    norm_X_sq = float(np.sum(np.array([np.linalg.norm(xx) ** 2.0 for xx in X_pf])))

    projected_X, norm_sq_err = project_data(
        X_pf, cp.zeros((1, 300)), cp_factors, norm_X_sq
    )

    np.testing.assert_allclose(norm_sq_err / norm_X_sq, 0.0, atol=1e-6)

    # De-mean since we aim to subtract off the means
    means = np.random.randn(X_pf[0].shape[1])  # noqa: NPY002
    X_pf = [xx + means for xx in X_pf]

    projected_X_mean, norm_sq_err_centered = project_data(
        X_pf, cp.array(means), cp_factors, norm_X_sq
    )

    for p_x, p_x_mean in zip(projected_X, projected_X_mean, strict=True):
        cp.testing.assert_allclose(p_x, p_x_mean, rtol=1.0e-4, atol=1.0e-4)
    np.testing.assert_allclose(
        norm_sq_err / norm_X_sq, norm_sq_err_centered / norm_X_sq, atol=1e-6
    )
