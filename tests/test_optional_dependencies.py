"""Installing AnyInit requires no framework, and using one requires only that one."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

import anyinit
from anyinit import backends

FRAMEWORKS = {"torch", "tensorflow", "keras", "jax", "jaxlib", "flax"}


def test_numpy_is_the_only_required_dependency():
    from importlib.metadata import requires

    required = [r for r in (requires("anyinit") or []) if "extra ==" not in r]
    assert [r.split(">")[0].split("=")[0].strip() for r in required] == ["numpy"]


def test_no_framework_extras_are_published():
    """There is one way to install AnyInit; frameworks are never its dependencies."""
    from importlib.metadata import metadata

    assert set(metadata("anyinit").get_all("Provides-Extra") or []) <= {"dev"}


def test_available_backends_are_a_subset_of_known_ones():
    assert set(anyinit.available_backends()) <= set(anyinit.known_backends())
    assert set(anyinit.known_backends()) == {"pytorch", "keras", "jax"}


def test_available_backends_reflect_what_is_installed():
    for cls in backends.known():
        expected = importlib.util.find_spec(cls.requires) is not None
        assert (cls.name in anyinit.available_backends()) == expected


def test_checking_availability_imports_no_framework():
    script = (
        "import sys, anyinit\n"
        "anyinit.available_backends(); anyinit.known_backends()\n"
        f"print(sorted({{m.split('.')[0] for m in sys.modules}} & {FRAMEWORKS!r}))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


def test_nothing_installed_says_how_to_install(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    assert anyinit.available_backends() == ()
    with pytest.raises(anyinit.BackendNotFoundError, match="no supported framework is installed"):
        anyinit.initialize(object())
    with pytest.raises(anyinit.BackendNotFoundError, match="picked up automatically"):
        anyinit.initialize(object())


def test_something_installed_lists_it():
    if not anyinit.available_backends():
        pytest.skip("no framework installed")
    with pytest.raises(anyinit.BackendNotFoundError, match="Installed backends"):
        anyinit.initialize(object())


def test_framework_free_features_work_without_a_backend():
    """Profiling and diagnosis are pure NumPy, so they need no framework at all."""
    assert anyinit.activation_profile("relu").chi == pytest.approx(1.0)
    anyinit.register_activation(lambda x: x * (x > 0) ** 2, name="t_opt_dep", overwrite=True)
    try:
        assert anyinit.activation_profile("t_opt_dep").moments(0.0, 1.0).m2 == pytest.approx(0.5)
    finally:
        anyinit.unregister_activation("t_opt_dep")
