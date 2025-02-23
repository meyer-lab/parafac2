"""
Test the data import.
"""

import pytest
import anndata
from scipy.sparse import csr_matrix
import numpy as np
import cupy as cp
from tensorly.decomposition import parafac2
from tensorly.random import random_parafac2
from ..parafac2 import parafac2_nd, parafac2_init
from ..utils import reconstruction_error, project_data
from tensorly.parafac2_tensor import parafac2_to_slices
from tensorly.decomposition._parafac2 import _parafac2_reconstruction_error


pf2shape = [(500, 2000)] * 8
X: list[np.ndarray] = random_parafac2(pf2shape, rank=3, full=True, random_state=2)  # type: ignore
norm_tensor = float(np.linalg.norm(X) ** 2)


def pf2_to_anndata(X_list, sparse=False):
    if sparse:
        X_list = [csr_matrix(XX) for XX in X_list]

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
    X_ann = pf2_to_anndata(X, sparse=sparse)

    f1, _ = parafac2_init(X_ann, rank=3, random_state=1)
    f2, _ = parafac2_init(X_ann, rank=3, random_state=1)

    # Compare both seeds
    for ii in range(3):
        cp.testing.assert_array_equal(f1[ii], f2[ii])

    proj1, proj_X1 = project_data(X, cp.zeros((1, X_ann.shape[1])), f1)
    proj2, proj_X2 = project_data(X, cp.zeros((1, X_ann.shape[1])), f2)

    # Compare both seeds
    cp.testing.assert_array_equal(proj_X1, proj_X2)

    for ii in range(len(proj1)):
        cp.testing.assert_array_equal(proj1[ii], proj2[ii])


@pytest.mark.parametrize("SECSI_solver", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_parafac2(sparse: bool, SECSI_solver: bool):
    """Test for equivalence to TensorLy's PARAFAC2."""
    X_ann = pf2_to_anndata(X, sparse=sparse)

    (w1, f1, p1), e1 = parafac2_nd(
        X_ann, rank=3, random_state=1, SECSI_solver=SECSI_solver
    )

    # Test that the model still matches the data
    err = _parafac2_reconstruction_error(X, (w1, f1, p1)) ** 2
    np.testing.assert_allclose(1.0 - err / norm_tensor, e1, rtol=1e-5)

    # Test reproducibility
    (w2, f2, p2), e2 = parafac2_nd(
        X_ann, rank=3, random_state=1, SECSI_solver=SECSI_solver
    )
    # Compare to TensorLy
    wT, fT, pT = parafac2(
        X,
        rank=3,
        normalize_factors=True,
        n_iter_max=5,
        init=(w1.copy(), [f.copy() for f in f1], [p.copy() for p in p1]),  # type: ignore
    )

    # Check normalization
    for ff in [f1, f2, fT]:
        for ii in range(3):
            np.testing.assert_allclose(np.linalg.norm(ff[ii], axis=0), 1.0, rtol=1e-2)

    # Compare both seeds
    np.testing.assert_allclose(w1, w2, rtol=1e-6)
    np.testing.assert_allclose(e1, e2)
    for ii in range(3):
        np.testing.assert_allclose(f1[ii], f2[ii], atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(p1[ii], p2[ii], atol=1e-8, rtol=1e-8)

    # Compare to TensorLy
    np.testing.assert_allclose(w1, wT, rtol=0.02)  # type: ignore
    for ii in range(3):
        np.testing.assert_allclose(f1[ii], fT[ii], rtol=0.01, atol=0.01)
        np.testing.assert_allclose(p1[ii], pT[ii], rtol=0.01, atol=0.01)


def test_pf2_r2x():
    """Compare R2X values to tensorly implementation"""
    w, f, _ = random_parafac2(pf2shape, rank=3, random_state=1, normalise_factors=False)

    p, projected_X = project_data(X, cp.zeros((1, X[0].shape[1])), f)
    errCMF = reconstruction_error(f, p, projected_X, norm_tensor)

    f = [cp.asnumpy(ff) for ff in f]
    p = [cp.asnumpy(pp) for pp in p]

    err = _parafac2_reconstruction_error(X, (w, f, p)) ** 2

    np.testing.assert_allclose(err, errCMF, rtol=1e-8)


@pytest.mark.parametrize("SECSI_solver", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("l1", [0.0, 0.00001])
def test_performance(sparse: bool, SECSI_solver: bool, l1: float):
    """Test for equivalence to TensorLy's PARAFAC2."""
    # 5000 by 2000 by 300 is roughly the lupus data
    pf2shape = [(5_00, 2_00)] * 30
    X = random_parafac2(pf2shape, rank=12, full=True, random_state=2)

    X = pf2_to_anndata(X, sparse=sparse)

    (w1, f1, p1), e1 = parafac2_nd(X, rank=9, SECSI_solver=SECSI_solver, l1=l1)


def test_pf2_proj_centering():
    """Test that centering the matrix does not affect the results."""
    _, factors, projections = random_parafac2(
        shapes=[(25, 300) for _ in range(15)],
        rank=3,
        normalise_factors=False,
        dtype=np.float64,
    )

    X_pf = parafac2_to_slices((None, factors, projections))

    norm_X_sq = float(np.sum(np.array([np.linalg.norm(xx) ** 2.0 for xx in X_pf])))  # type: ignore

    projections, projected_X = project_data(X_pf, cp.zeros((1, 300)), factors)
    norm_sq_err = reconstruction_error(factors, projections, projected_X, norm_X_sq)

    np.testing.assert_allclose(norm_sq_err / norm_X_sq, 0.0, atol=1e-6)

    # De-mean since we aim to subtract off the means
    means = np.random.randn(X_pf[0].shape[1])
    X_pf = [xx + means for xx in X_pf]

    projections, projected_X_mean = project_data(X_pf, cp.array(means), factors)
    norm_sq_err_centered = reconstruction_error(
        factors, projections, projected_X, norm_X_sq
    )

    cp.testing.assert_allclose(projected_X, projected_X_mean)
    np.testing.assert_allclose(
        norm_sq_err / norm_X_sq, norm_sq_err_centered / norm_X_sq, atol=1e-6
    )
