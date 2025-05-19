import numpy as np
import tensorly as tl
from tensorly.metrics.similarity import correlation_index as corridx
from tensorly.random import random_cp

from ..SECSI import SECSI


def SECSItest(dim, true_rank, est_rank, noise=0.0, verbose=True):
    """
    Built to test SECSI.py function. Creates three random factor matrices based
    on given dimension, rank.
    Computes CP tensor based on these matrices.
    Optionally adds noise to tensor. Tensor is then fed to SECSI,
    which outputs estimated factor matrices.
    Estimate tensor is built from these estimated factor matrices
    All estimates are evaluated, ranked by accuracy.

    Args:
    dim: tuple with desired tensor dimensions
    true_rank: true rank of the input tensor, length of randomized factor matrices
    est_rank: rank with which to compute estimate factor matrices
    noise: multiple of gaussian noise to be added to tensor
    random_state, for consistency.

    """
    tensor_fac: tl.cp_tensor.CPTensor = random_cp(dim, true_rank, full=False)  # type: ignore
    tensor = tl.cp_to_tensor(tensor_fac)

    # Adds noise
    tensor = tensor + np.random.normal(size=dim, scale=noise)  # noqa: NPY002

    norm_est, cp_estimates = SECSI(tensor, est_rank, 50, verbose=False)

    for cp_est in cp_estimates:
        assert (
            float(corridx(tensor_fac.factors, cp_est.factors, method="min_score"))
            < 0.01
        )

    if verbose:
        for i, resid in enumerate(norm_est):
            if i == np.argmin(norm_est):
                print(f"Best estimate, {i}, has error: {resid:.3e}")
            else:
                print(f"Estimate #{i} has error: {resid:.3e}")
    return np.min(norm_est)


def test_SECSI():
    SECSItest((60, 50, 40), 10, 10, noise=0.0)
