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
from ..parafac2 import parafac2_nd
from ..utils import reconstruction_error, project_data
from tensorly.decomposition._parafac2 import _parafac2_reconstruction_error


pf2shape = [(100, 800)] * 4
X = random_parafac2(pf2shape, rank=3, full=True, random_state=2)
norm_tensor = np.linalg.norm(X) ** 2  # type: ignore


def pf2_to_anndata(X_list, sparse=False):
    if sparse:
        X_list = [csr_matrix(XX) for XX in X_list]

    X_ann = [anndata.AnnData(XX) for XX in X_list]

    X_merged = anndata.concat(
        X_ann, label="condition_unique_idxs", keys=np.arange(len(X_list))
    )
    X_merged.var["means"] = np.zeros(X_list[0].shape[1])

    return X_merged


@pytest.mark.parametrize("sparse", [False, True])
def test_parafac2(sparse: bool):
    """Test for equivalence to TensorLy's PARAFAC2."""
    X_ann = pf2_to_anndata(X, sparse=sparse)

    (w1, f1, p1), e1 = parafac2_nd(X_ann, rank=3, random_state=1)

    # Test that the model still matches the data
    err = _parafac2_reconstruction_error(X, (w1, f1, p1)) ** 2
    np.testing.assert_allclose(1.0 - err / norm_tensor, e1, rtol=1e-5)

    # Test reproducibility
    (w2, f2, p2), e2 = parafac2_nd(X_ann, rank=3, random_state=1)
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
        np.testing.assert_allclose(f1[ii], f2[ii], atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(p1[ii], p2[ii], atol=1e-6, rtol=1e-5)

    # Compare to TensorLy
    np.testing.assert_allclose(w1, wT, rtol=0.02)  # type: ignore
    for ii in range(3):
        np.testing.assert_allclose(f1[ii], fT[ii], rtol=0.01, atol=0.01)
        np.testing.assert_allclose(p1[ii], pT[ii], rtol=0.01, atol=0.01)


@pytest.mark.parametrize("sparse", [False, True])
def test_pf2_r2x(sparse: bool):
    """Compare R2X values to tensorly implementation"""
    w, f, _ = random_parafac2(pf2shape, rank=3, random_state=1, normalise_factors=False)

    X_ann = pf2_to_anndata(X, sparse=sparse)
    p, projected_X = project_data(X_ann, f)
    errCMF = reconstruction_error(f, p, cp.asnumpy(projected_X), norm_tensor)

    err = _parafac2_reconstruction_error(X, (w, f, p)) ** 2

    np.testing.assert_allclose(err, errCMF, rtol=1e-8)


@pytest.mark.parametrize("sparse", [False, True])
def test_performance(sparse: bool):
    """Test for equivalence to TensorLy's PARAFAC2."""
    # 5000 by 2000 by 300 is roughly the lupus data
    pf2shape = [(5000, 2000)] * 10
    X = random_parafac2(pf2shape, rank=12, full=True, random_state=2)

    X = pf2_to_anndata(X, sparse=sparse)

    (w1, f1, p1), e1 = parafac2_nd(X, rank=9)
