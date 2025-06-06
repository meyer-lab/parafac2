"""
Test the data import.
"""

import anndata
import cupy as cp
import numpy as np
import pytest
from scipy.sparse import csr_array
from tensorly.decomposition import parafac2
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
    X_reprod: list[np.ndarray] = random_parafac2(pf2shape_reprod, rank=3, full=True)  # type: ignore

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

    proj_X1, _ = project_data(X_reprod, means, f1, 1.0)
    proj_X2, _ = project_data(X_reprod, means, f2, 1.0)

    # Compare both seeds
    cp.testing.assert_array_equal(proj_X1, proj_X2)


@pytest.mark.parametrize("sparse", [False, True])
def test_parafac2(sparse: bool):
    """Test for equivalence to TensorLy's PARAFAC2."""
    # 5000 by 2000 by 300 is roughly the lupus data
    pf2shape = [(200, 500)] * 6
    X: list[np.ndarray] = random_parafac2(pf2shape, rank=3, full=True, random_state=2)  # type: ignore
    norm_tensor = float(np.linalg.norm(X) ** 2)

    X_ann = pf2_to_anndata(X, sparse=sparse)

    options = {"tol": 1e-12, "n_iter_max": 1000}

    (w1, f1, p1), e1 = parafac2_nd(X_ann, rank=3, random_state=1, **options)

    # Test that the model still matches the data
    err = _parafac2_reconstruction_error(X, (w1, f1, p1)) ** 2
    assert err / norm_tensor < 5e-4
    assert (1.0 - e1) < 5e-4

    # Test reproducibility
    (w2, f2, p2), e2 = parafac2_nd(X_ann, rank=3, random_state=3, **options)

    # Compare to TensorLy
    wT, fT, pT = parafac2(  # type: ignore
        X,
        rank=3,
        normalize_factors=True,
        n_iter_max=10,
        init=(w1.copy(), [f.copy() for f in f1], [p.copy() for p in p1]),  # type: ignore
    )

    # Check normalization
    for ff in [f1, f2, fT]:
        for ii in range(3):
            np.testing.assert_allclose(np.linalg.norm(ff[ii], axis=0), 1.0, rtol=1e-3)

    # Compare both seeds
    np.testing.assert_allclose(w1, w2, rtol=0.02)
    np.testing.assert_allclose(e1, e2, rtol=1e-4)
    for ii in range(3):
        np.testing.assert_allclose(f1[ii], f2[ii], atol=1e-2, rtol=1e-2)
        np.testing.assert_allclose(p1[ii], p2[ii], atol=1e-2, rtol=1e-2)

    # Compare to TensorLy
    np.testing.assert_allclose(w1, wT, rtol=0.2)  # type: ignore
    for ii in range(3):
        np.testing.assert_allclose(f1[ii], fT[ii], rtol=0.01, atol=0.05)
        np.testing.assert_allclose(p1[ii], pT[ii], rtol=0.01, atol=0.05)


def test_pf2_r2x():
    """Compare R2X values to tensorly implementation"""
    pf2shape = [(50, 200)] * 8
    X: list[np.ndarray] = random_parafac2(pf2shape, rank=3, full=True, random_state=2)  # type: ignore
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

    norm_X_sq = float(np.sum(np.array([np.linalg.norm(xx) ** 2.0 for xx in X_pf])))  # type: ignore

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

    cp.testing.assert_allclose(projected_X, projected_X_mean, atol=1.0e-4)  # type: ignore
    np.testing.assert_allclose(
        norm_sq_err / norm_X_sq, norm_sq_err_centered / norm_X_sq, atol=1e-6
    )
