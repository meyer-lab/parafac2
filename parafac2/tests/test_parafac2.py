"""
Tests for the core PARAFAC2 fitting routines (``parafac2`` module), the
supporting numerical utilities (``utils`` module), and the compute backend
abstraction (``backend`` module).
"""

import anndata
import numpy as np
import pytest
from scipy.sparse import csr_array
from tensorly.decomposition._parafac2 import _parafac2_reconstruction_error
from tensorly.parafac2_tensor import parafac2_to_slices
from tensorly.random import random_parafac2

from ..parafac2 import parafac2_init, parafac2_nd
from ..utils import calc_norm_sq, calc_slice_norms, project_data


def pf2_to_anndata(X_list, sparse=False):
    """Build a single concatenated AnnData object from a list of per-condition matrices.

    Parameters
    ----------
    X_list : list[np.ndarray]
        Per-condition data matrices, each with shape ``(n_cells_k, n_genes)``.
    sparse : bool, default False
        Whether to convert each matrix to a ``csr_array`` before wrapping it.

    Returns
    -------
    anndata.AnnData
        The concatenated dataset, with ``obs["condition_unique_idxs"]`` set
        to the source index of each row and ``var["means"]`` zeroed.
    """
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
        """Record each iteration's relative error for the monotonicity check below."""
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


def test_calc_slice_norms():
    """Test calc_slice_norms against a brute-force per-condition reference,
    for dense and sparse input, with and without means."""
    rng = np.random.default_rng(0)
    shapes = [10, 15, 7]
    n_cond = len(shapes)
    X_list = [rng.normal(size=(n, 12)) for n in shapes]
    X_dense = np.concatenate(X_list, axis=0)
    cond_idxs = np.concatenate([[i] * n for i, n in enumerate(shapes)])

    # Introduce sparsity for the sparse variant
    X_sparse_dense = X_dense.copy()
    X_sparse_dense[rng.random(X_sparse_dense.shape) > 0.5] = 0.0
    X_sparse = csr_array(X_sparse_dense)

    for means in (None, rng.normal(size=12)):
        expected = np.array(
            [
                np.linalg.norm(
                    X_dense[cond_idxs == i] - (means if means is not None else 0.0)
                )
                for i in range(n_cond)
            ]
        )
        result_dense = calc_slice_norms(X_dense, means, cond_idxs, n_cond)
        np.testing.assert_allclose(result_dense, expected, rtol=1e-10)

        expected_sparse = np.array(
            [
                np.linalg.norm(
                    X_sparse_dense[cond_idxs == i]
                    - (means if means is not None else 0.0)
                )
                for i in range(n_cond)
            ]
        )
        result_sparse = calc_slice_norms(X_sparse, means, cond_idxs, n_cond)
        np.testing.assert_allclose(result_sparse, expected_sparse, rtol=1e-8, atol=1e-8)


def test_parafac2_normalize_slices_runs_and_orthonormal():
    """normalize_slices=True should run to completion and still yield valid,
    orthonormal projections and a sane R2X."""
    rng = np.random.default_rng(7)
    # Deliberately imbalanced condition sizes/scales so the option matters.
    shapes = [(80, 20), (10, 20), (10, 20)]
    rank = 3

    X_list = [rng.normal(size=shape) for shape in shapes]
    X_list[0] *= 5.0  # first condition dominates in scale as well as count
    X_ann = pf2_to_anndata(X_list, sparse=False)

    (_w, _f, p), r2x = parafac2_nd(
        X_ann,
        rank=rank,
        random_state=7,
        n_iter_max=50,
        tol=1e-6,
        normalize_slices=True,
    )

    assert 0.0 <= r2x <= 1.0
    for P_k in p:
        np.testing.assert_allclose(P_k.T @ P_k, np.eye(rank), atol=1e-5)


def test_parafac2_normalize_slices_changes_result():
    """normalize_slices=True should give a different fit than the default
    when condition scales/sizes are highly imbalanced."""
    rng = np.random.default_rng(11)
    shapes = [(100, 15), (8, 15), (8, 15)]
    rank = 2

    X_list = [rng.normal(size=shape) for shape in shapes]
    X_list[0] *= 8.0
    X_ann = pf2_to_anndata(X_list, sparse=False)

    (_w_off, f_off, _p_off), _r2x_off = parafac2_nd(
        X_ann, rank=rank, random_state=11, n_iter_max=50, tol=1e-6
    )
    (_w_on, f_on, _p_on), _r2x_on = parafac2_nd(
        X_ann,
        rank=rank,
        random_state=11,
        n_iter_max=50,
        tol=1e-6,
        normalize_slices=True,
    )

    # The two runs should not be numerically identical: the weighting
    # changes which factors the MTTKRP updates converge to.
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(f_off[0], f_on[0], rtol=1e-4, atol=1e-4)


def _check_backend_available(backend: str) -> bool:
    """Return whether the given compute backend's package is importable.

    Parameters
    ----------
    backend : str
        One of ``'cpu'``, ``'mlx'``, or ``'cupy'``.

    Returns
    -------
    bool
        True if the backend can be used in this environment.
    """
    if backend == "cpu":
        return True
    elif backend == "mlx":
        try:
            import mlx.core  # noqa: F401  # ty: ignore[unresolved-import]

            return True
        except ImportError:
            return False
    elif backend == "cupy":
        try:
            import cupy  # noqa: F401  # ty: ignore[unresolved-import]

            return True
        except ImportError:
            return False
    return False


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("backend", ["cpu", "mlx", "cupy"])
def test_backend_matrix_ops(sparse: bool, backend: str):
    """Test GPUMatrix matmul and rmatmul for available backends against NumPy."""
    from ..backend import GPUMatrix

    if not _check_backend_available(backend):
        pytest.skip(f"Backend '{backend}' is not installed.")

    rng = np.random.default_rng(42)
    raw = rng.normal(size=(25, 20)).astype(np.float32)
    mat = csr_array(raw) if sparse else raw

    gpu_mat = GPUMatrix(mat, backend=backend)

    # Left matmul (2D and 1D)
    rhs2 = rng.normal(size=(20, 5)).astype(np.float32)
    res_mat = gpu_mat @ rhs2
    expected_mat = raw @ rhs2
    np.testing.assert_allclose(res_mat, expected_mat, rtol=1e-5, atol=1e-5)

    # Right matmul (2D and 1D)
    lhs2 = rng.normal(size=(4, 25)).astype(np.float32)
    res_rmat = lhs2 @ gpu_mat
    expected_rmat = lhs2 @ raw
    np.testing.assert_allclose(res_rmat, expected_rmat, rtol=1e-5, atol=1e-5)


def test_invalid_backend():
    """Test that get_backend raises ValueError for an unrecognized backend name."""
    from ..backend import get_backend

    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("nonexistent_backend")


def test_get_backend_fallback(monkeypatch):
    """Test that get_backend falls back to 'cpu' when mlx and cupy are unimportable."""
    from ..backend import get_backend

    monkeypatch.setattr(
        "builtins.__import__",
        lambda name, *args, **kwargs: (
            (_ for _ in ()).throw(ImportError)
            if name in ("mlx.core", "cupy")
            else __import__(name, *args, **kwargs)
        ),
    )
    assert get_backend() == "cpu"
