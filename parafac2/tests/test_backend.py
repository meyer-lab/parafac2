"""Tests for backend resolution."""

import sys
import types

import numpy as np
import pytest
from scipy.sparse import csr_array

from parafac2 import backend as backend_mod
from parafac2.backend import (
    BACKEND_ENV_VAR,
    _cuda_is_usable,
    _ensure_device_capacity,
    device_bytes,
    get_backend,
)


def test_explicit_backend_is_returned_verbatim():
    for name in ("cpu", "cupy", "mlx"):
        assert get_backend(name) == name


def test_explicit_backend_is_case_and_whitespace_insensitive():
    assert get_backend(" CuPy ") == "cupy"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("cuda")


def test_env_var_forces_a_backend(monkeypatch):
    """`PARAFAC2_BACKEND=cpu` is the supported way to force the CPU path.

    It is the only route on a machine where CuPy imports but the data does not
    fit on the device: `CUDA_VISIBLE_DEVICES=""` does not prevent the import.
    """
    monkeypatch.setenv(BACKEND_ENV_VAR, "cpu")
    assert get_backend() == "cpu"


def test_explicit_argument_beats_the_env_var(monkeypatch):
    monkeypatch.setenv(BACKEND_ENV_VAR, "cpu")
    assert get_backend("mlx") == "mlx"


def test_unknown_env_var_raises_and_names_its_source(monkeypatch):
    monkeypatch.setenv(BACKEND_ENV_VAR, "gpu")
    with pytest.raises(ValueError, match=BACKEND_ENV_VAR):
        get_backend()


def test_empty_env_var_falls_through_to_autodetection(monkeypatch):
    monkeypatch.setenv(BACKEND_ENV_VAR, "")
    assert get_backend() in ("mlx", "cupy", "cpu")


def test_autodetection_requires_a_real_cuda_device(monkeypatch):
    """An importable CuPy with no visible device must not select 'cupy'.

    Regression test: auto-detection previously returned "cupy" whenever the
    module imported, so a machine with CuPy installed but no usable GPU failed
    at the first allocation instead of running on the CPU.
    """
    monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)
    monkeypatch.setattr(backend_mod, "_cuda_is_usable", lambda: False)
    assert get_backend() != "cupy"


def test_cuda_probe_survives_a_broken_driver(monkeypatch):
    """A driver/runtime error must degrade to False rather than propagate."""

    def _raise():
        raise RuntimeError("CUDA driver version is insufficient")

    fake = types.ModuleType("cupy")
    fake.cuda = types.SimpleNamespace(  # ty: ignore[unresolved-attribute]
        runtime=types.SimpleNamespace(getDeviceCount=_raise)
    )
    monkeypatch.setitem(sys.modules, "cupy", fake)
    assert _cuda_is_usable() is False


def test_cuda_probe_reports_no_device_as_false(monkeypatch):
    fake = types.ModuleType("cupy")
    fake.cuda = types.SimpleNamespace(  # ty: ignore[unresolved-attribute]
        runtime=types.SimpleNamespace(getDeviceCount=lambda: 0)
    )
    monkeypatch.setitem(sys.modules, "cupy", fake)
    assert _cuda_is_usable() is False


def test_cuda_probe_returns_a_bool():
    assert isinstance(_cuda_is_usable(), bool)


# ---------------------------------------------------------------------------
# Device capacity accounting
# ---------------------------------------------------------------------------


def test_device_bytes_dense_is_the_array_size():
    a = np.zeros((100, 50), dtype=np.float32)
    assert device_bytes(a) == a.nbytes


def test_device_bytes_sparse_counts_a_shared_int32_index_dtype():
    """Small matrices index in int32, for both `indices` and `indptr`."""
    rows, cols, nnz = 100, 40, 500
    rng = np.random.default_rng(0)
    m = csr_array(
        (
            rng.random(nnz, dtype=np.float32),
            np.sort(rng.integers(0, cols, size=nnz)).astype(np.int32),
            np.linspace(0, nnz, rows + 1).astype(np.int32),
        ),
        shape=(rows, cols),
    )
    assert device_bytes(m) == m.data.nbytes + (nnz + rows + 1) * 4


def test_device_bytes_charges_int64_once_nnz_exceeds_int32(monkeypatch):
    """A matrix with >2**31 nonzeros pays 8 bytes per column index.

    `cupyx` stores `indices` and `indptr` in one shared index dtype, so a huge
    nonzero count widens the *column indices* too even though their values
    would fit in 4 bytes. This is the whole reason a cohort-scale matrix is so
    much larger on the device than in host memory, so the estimate has to model
    it rather than trusting the host arrays' own dtypes.
    """

    class _Huge:
        """Stands in for a >2**31-nnz CSR without allocating one."""

        shape = (1_185_861, 23_729)

        class data:
            size = 3_600_000_000
            nbytes = 3_600_000_000 * 4

    monkeypatch.setattr(backend_mod, "issparse", lambda _m: True)
    got = device_bytes(_Huge)  # ty: ignore[invalid-argument-type]
    expected = _Huge.data.nbytes + (3_600_000_000 + 1_185_861 + 1) * 8
    assert got == expected
    # int32 accounting would have understated it by ~14.5 GB.
    assert got - (_Huge.data.nbytes + (3_600_000_000 + 1_185_862) * 4) > 14e9


def test_capacity_check_is_a_noop_when_the_matrix_fits(monkeypatch):
    monkeypatch.setattr(backend_mod, "_managed_allocator_installed", False)
    fake_cupy = _fake_cupy(free=100_000, total=100_000)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    assert _ensure_device_capacity(1_000) is False
    assert fake_cupy.cuda.set_allocator.calls == []


def test_capacity_check_installs_managed_memory_when_it_does_not_fit(monkeypatch):
    monkeypatch.setattr(backend_mod, "_managed_allocator_installed", False)
    fake_cupy = _fake_cupy(free=1_000, total=2_000, managed=True)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    assert _ensure_device_capacity(10_000) is True
    assert len(fake_cupy.cuda.set_allocator.calls) == 1
    assert backend_mod._managed_allocator_installed is True


def test_capacity_check_raises_a_useful_error_when_it_cannot_oversubscribe(
    monkeypatch,
):
    """No managed-memory support and no room: fail clearly, not with an OOM."""
    monkeypatch.setattr(backend_mod, "_managed_allocator_installed", False)
    monkeypatch.setitem(
        sys.modules, "cupy", _fake_cupy(free=1_000, total=2_000, managed=False)
    )
    with pytest.raises(MemoryError, match=BACKEND_ENV_VAR):
        _ensure_device_capacity(10_000)


def test_capacity_check_does_not_reinstall_the_pool(monkeypatch):
    """CuPy's allocator is process-global; installing twice discards the first."""
    monkeypatch.setattr(backend_mod, "_managed_allocator_installed", True)
    fake_cupy = _fake_cupy(free=1_000, total=2_000, managed=True)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    assert _ensure_device_capacity(10_000) is True
    assert fake_cupy.cuda.set_allocator.calls == []


def _fake_cupy(free: int, total: int, managed: bool = True):
    """A stand-in for `cupy` exposing only what the capacity check touches."""

    class _Recorder:
        def __init__(self):
            self.calls = []

        def __call__(self, *args):
            self.calls.append(args)

    props = {
        "managedMemory": int(managed),
        "concurrentManagedAccess": int(managed),
    }

    fake = types.ModuleType("cupy")
    fake.cuda = types.SimpleNamespace(  # ty: ignore[unresolved-attribute]
        Device=lambda: types.SimpleNamespace(mem_info=(free, total), id=0),
        runtime=types.SimpleNamespace(getDeviceProperties=lambda _i: props),
        set_allocator=_Recorder(),
        MemoryPool=lambda alloc: types.SimpleNamespace(malloc=alloc),
        malloc_managed=object(),
    )
    return fake
