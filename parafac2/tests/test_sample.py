import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ..sample import SampleArray, sample_array


def test_sample_array_alias():
    assert sample_array is SampleArray


@pytest.mark.parametrize("sparse", [False, True])
def test_sample_array_matmul(sparse: bool):
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(20, 15)).astype(np.float32)
    means = rng.normal(size=15).astype(np.float32)
    centered = raw - means

    mat = csr_matrix(raw) if sparse else raw
    sa = SampleArray(mat, means)

    assert sa.shape == (20, 15)
    assert len(sa) == 20
    assert sa.ndim == 2

    # Test toarray
    np.testing.assert_allclose(sa.toarray(), centered, rtol=1e-5, atol=1e-5)

    # Test left matmul (2D and 1D)
    rhs2 = rng.normal(size=(15, 5)).astype(np.float32)
    np.testing.assert_allclose(sa @ rhs2, centered @ rhs2, rtol=1e-5, atol=1e-5)

    rhs1 = rng.normal(size=15).astype(np.float32)
    np.testing.assert_allclose(sa @ rhs1, centered @ rhs1, rtol=1e-5, atol=1e-5)

    # Test right matmul (2D and 1D)
    lhs2 = rng.normal(size=(4, 20)).astype(np.float32)
    np.testing.assert_allclose(lhs2 @ sa, lhs2 @ centered, rtol=1e-5, atol=1e-5)

    lhs1 = rng.normal(size=20).astype(np.float32)
    np.testing.assert_allclose(lhs1 @ sa, lhs1 @ centered, rtol=1e-5, atol=1e-5)
