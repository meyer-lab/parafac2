import numpy as np
import pytest
from scipy.sparse import csr_array

from ..sample import SampleArray


def _check_backend_available(backend: str) -> bool:
    if backend == "cpu":
        return True
    elif backend == "mlx":
        try:
            import mlx.core  # noqa: F401

            return True
        except ImportError:
            return False
    elif backend == "cupy":
        try:
            import cupy  # noqa: F401

            return True
        except ImportError:
            return False
    return False


@pytest.mark.parametrize("sparse", [False, True])
def test_sample_array_matmul(sparse: bool):
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(20, 15)).astype(np.float32)
    means = rng.normal(size=15).astype(np.float32)
    centered = raw - means

    mat = csr_array(raw) if sparse else raw
    sa = SampleArray(mat, means)

    assert sa.shape == (20, 15)
    assert len(sa) == 20
    assert sa.ndim == 2

    # Test toarray and norm_sq
    np.testing.assert_allclose(sa.toarray(), centered, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sa.norm_sq(), np.sum(centered**2), rtol=1e-5, atol=1e-5)

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


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("backend", ["cpu", "mlx", "cupy"])
def test_backend_equivalence(sparse: bool, backend: str):
    if not _check_backend_available(backend):
        pytest.skip(f"Backend '{backend}' is not installed.")

    rng = np.random.default_rng(42)
    raw = rng.normal(size=(20, 15)).astype(np.float32)
    means = rng.normal(size=15).astype(np.float32)
    centered = raw - means

    mat = csr_array(raw) if sparse else raw
    sa = SampleArray(mat, means)

    rhs2 = rng.normal(size=(15, 5)).astype(np.float32)
    rhs1 = rng.normal(size=15).astype(np.float32)
    lhs2 = rng.normal(size=(4, 20)).astype(np.float32)
    lhs1 = rng.normal(size=20).astype(np.float32)

    res_matmul2 = sa.__matmul__(rhs2, backend=backend)
    res_matmul1 = sa.__matmul__(rhs1, backend=backend)
    res_rmatmul2 = sa.__rmatmul__(lhs2, backend=backend)
    res_rmatmul1 = sa.__rmatmul__(lhs1, backend=backend)

    # Verify against CPU centered reference
    np.testing.assert_allclose(res_matmul2, centered @ rhs2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_matmul1, centered @ rhs1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_rmatmul2, lhs2 @ centered, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_rmatmul1, lhs1 @ centered, rtol=1e-5, atol=1e-5)

    # Compare directly against CPU backend output
    cpu_matmul2 = sa.__matmul__(rhs2, backend="cpu")
    cpu_matmul1 = sa.__matmul__(rhs1, backend="cpu")
    cpu_rmatmul2 = sa.__rmatmul__(lhs2, backend="cpu")
    cpu_rmatmul1 = sa.__rmatmul__(lhs1, backend="cpu")

    np.testing.assert_allclose(res_matmul2, cpu_matmul2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_matmul1, cpu_matmul1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_rmatmul2, cpu_rmatmul2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_rmatmul1, cpu_rmatmul1, rtol=1e-5, atol=1e-5)


def test_invalid_backend():
    from ..sample import get_backend

    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("invalid_backend_name")


def test_sample_array_dtype():
    raw = np.ones((5, 5), dtype=np.float32)
    means = np.zeros(5, dtype=np.float32)
    sa = SampleArray(raw, means)
    assert sa.dtype == np.float32


def test_get_backend_cpu_fallback(monkeypatch):
    from ..sample import get_backend

    # Monkeypatch mlx and cupy imports to fail
    monkeypatch.setattr(
        "builtins.__import__",
        lambda name, *args, **kwargs: (
            (_ for _ in ()).throw(ImportError)
            if name in ("mlx.core", "cupy")
            else __import__(name, *args, **kwargs)
        ),
    )
    assert get_backend() == "cpu"
