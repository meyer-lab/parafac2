"""Tests for backend resolution."""

import sys
import types

import pytest

from parafac2 import backend as backend_mod
from parafac2.backend import BACKEND_ENV_VAR, _cuda_is_usable, get_backend


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
