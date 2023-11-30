import numpy as np
import pytest
from tensorly.metrics.factors import congruence_coefficient

from parafac2.pf2_plsr import *


DEFAULT_IJK = (10, 10, 10)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_factor_normality(rank):
    (X, y), _, _ = gen_synthetic_dataset(rank, *DEFAULT_IJK)
    pls = PF2_PLSR(rank).fit(X, y)
    np.testing.assert_allclose(np.linalg.norm(pls.Omega_J, axis=2), 1)
    np.testing.assert_allclose(np.linalg.norm(pls.Omega_K, axis=1), 1)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_decomposition_accuracy(rank):
    """Test that PF2_PLSR recovers factors in original synthetic data."""
    (X, y), (Omega_J_init, Omega_K_init, T_init), _ = gen_synthetic_dataset(
        rank, *DEFAULT_IJK, y_mixed_comps=True
    )
    pls = PF2_PLSR(rank).fit(X, y, init="ones", center=True)

    assert (
        congruence_coefficient(
            pls.Omega_J.reshape(rank, -1), Omega_J_init.reshape(rank, -1)
        )[0]
        > 0.8
    )
    assert (
        congruence_coefficient(
            pls.Omega_K.reshape(rank, -1), Omega_K_init.reshape(rank, -1)
        )[0]
        > 0.75
    )
    assert (
        congruence_coefficient(pls.T.reshape(rank, -1), T_init.reshape(rank, -1))[0]
        > 0.75
    )


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_reconstruction_X(rank):
    """Tests that reconstructing X from its factors yields something close to
    original data."""
    (X, y), _, _ = gen_synthetic_dataset(rank, *DEFAULT_IJK)
    pls = PF2_PLSR(rank)
    pls.fit(X, y, max_iter=15000, tol=1e-15, center=False)
    X_reconstructed = pls.reconstruct_X()
    X = X_slices_to_tensor(X)
    np.testing.assert_allclose(X_reconstructed, X, rtol=1e-2)


def test_reconstruction_X_padded():
    """Same as test_reconstruction_X except the dataset is composed of slices
    of varying J_i, plus the synthetic data is not known to be of particular
    rank."""
    K = 5
    I = 5
    Ji = np.random.choice(np.arange(10, 20), I)
    J = np.max(Ji)
    I = len(Ji)

    X = [np.random.rand(Ji[i], K) for i in range(I)]
    y = np.random.rand(I)

    X = center_X(X)

    pls = PF2_PLSR(5)
    pls.fit(X, y)

    np.testing.assert_allclose(X_slices_to_tensor(X), pls.reconstruct_X(), rtol=1e-1)


def test_rand_covariance():
    """Tests PF2_PLSR captures covariance between random, unrelated x and y"""
    I, J, K = DEFAULT_IJK
    rank = 2
    X = np.random.rand(I, J, K)
    y = np.random.rand(I)
    pls = PF2_PLSR(rank)
    pls.fit(X, y)
    for r in range(rank):
        cov = np.corrcoef(pls.T[r], y)[0, 1]
        assert cov > 0.0 and cov < 1.0


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_maximum_covariance(rank):
    """Tests CP_PLSR components capture maximum covariance in synthetic data."""
    (X, y), (_, _, T_init), b_init = gen_synthetic_dataset(
        rank, *DEFAULT_IJK, y_mixed_comps=False
    )

    pls = PF2_PLSR(rank)
    pls.fit(X, y)

    assert np.isclose(
        np.corrcoef(pls.T.T @ pls.b, y)[0, 1],
        1,
        atol=5e-2,
    )


@pytest.mark.parametrize("rank", [1, 2])
def test_prediction(rank):
    """Tests that the y generated from fitting is close to the original y."""
    (X, y), _, _ = gen_synthetic_dataset(rank, *DEFAULT_IJK, y_mixed_comps=False)
    pls = PF2_PLSR(rank)
    pls.fit(X, y, center=False)
    np.testing.assert_allclose(pls.y_fit, y, rtol=1e-1)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_prediction_mixed_comps(rank):
    """Same as test_prediction except y is composed as the combination of
    multiple components in X."""
    (X, y), _, _ = gen_synthetic_dataset(rank, *DEFAULT_IJK, y_mixed_comps=True)
    pls = PF2_PLSR(rank)
    pls.fit(X, y, center=False)
    assert congruence_coefficient(pls.y_fit[None, :], y[None, :])[0] > 0.95


def gen_synthetic_dataset(rank, I, J, K, y_mixed_comps=False):
    """
    Generates synthetic dataset of fixed rank for testing. Slices along I
    dimension will have the same value for J_i, so they are aligned.

    Args:
        rank: rank of synthetic dataset.
        I, J, K: dimensions of dataset.
        y_mixed_comps: designates whether y should be composed of the magnitude
          of one component (False) or linear combination of multiple components
          (True).

    Returns:
        ((X, y), (Omega_J, Omega_K, T), b)
        where Omega_J are the J dimension factors, Omega_K are the K dimension
        factors, T are the magnitudes of the factors, and b are the weights for
        transforming T into y
    """
    # generate orthonormal factors
    assert rank < J, "rank < J for factors to be orthogonal"
    Omega_J_slices = [np.random.uniform(-1, 1, (J, rank)) for i in range(I)]
    Omega_J_slices = [np.linalg.qr(slice)[0] for slice in Omega_J_slices]
    Omega_J = np.array(Omega_J_slices)
    Omega_J = np.transpose(Omega_J, axes=(2, 0, 1))
    assert Omega_J.shape == (rank, I, J)
    if rank > 1:
        assert are_orthogonal(Omega_J[0, 0], Omega_J[1, 0])
    np.testing.assert_allclose(np.linalg.norm(Omega_J, axis=2), 1)

    Omega_K = np.linalg.qr(np.random.uniform(-1, 1, (K, rank)))[0].T
    if rank > 1:
        assert are_orthogonal(Omega_K[0], Omega_K[1])
    np.testing.assert_allclose(np.linalg.norm(Omega_K, axis=1), 1)

    T = np.linalg.qr(np.random.uniform(-1, 1, (I, rank)))[0].T

    X = reconstruct_X(Omega_J, Omega_K, T)
    if y_mixed_comps:
        b = np.random.uniform(-1, 1, rank)
    else:
        b = np.zeros(rank)
        b[0] = 1
    y = T.T @ b

    return (X_tensor_to_slices(X), y), (Omega_J, Omega_K, T), b


def are_orthogonal(v1, v2):
    return np.isclose(np.dot(v1, v2), 0, rtol=1e-15)
