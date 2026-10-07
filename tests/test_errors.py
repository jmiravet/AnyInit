"""Failure paths of the public API and backend resolution."""

from __future__ import annotations

import numpy as np
import pytest

import anyinit
from anyinit import backends
from anyinit.errors import BackendNotFoundError, BackendUnavailableError, ConfigError


def test_a_backend_whose_check_raises_is_skipped(monkeypatch):
    def broken(model):
        raise RuntimeError("probe failed")

    for cls in backends.known():
        monkeypatch.setattr(cls, "handles", staticmethod(broken))
    with pytest.raises(BackendNotFoundError):
        backends.resolve(object())


def test_a_recognised_model_whose_framework_will_not_import_is_explained(monkeypatch):
    cls = backends.known()[0]

    def refuse(self):
        raise ImportError("simulated")

    monkeypatch.setattr(cls, "handles", staticmethod(lambda model: True))
    monkeypatch.setattr(cls, "__init__", refuse)
    with pytest.raises(BackendUnavailableError, match=rf"pip install {cls.install}"):
        backends.resolve(object())


def test_non_callable_activation_is_refused():
    with pytest.raises(ConfigError, match="must be callable"):
        anyinit.register_activation(3.0, name="not_a_function")


def test_duplicate_registration_needs_overwrite():
    def twice(x):
        return np.maximum(x, 0.0) * 2.0

    anyinit.register_activation(twice)
    try:
        with pytest.raises(Exception, match="twice"):
            anyinit.register_activation(twice)
        anyinit.register_activation(twice, overwrite=True)
    finally:
        anyinit.unregister_activation("twice")


def test_decorator_form_with_arguments_registers_under_the_given_name():
    @anyinit.register_activation(name="half_relu")
    def anything(x):
        return 0.5 * np.maximum(x, 0.0)

    try:
        assert "half_relu" in anyinit.registered_activations()
        assert anyinit.activation_profile("half_relu").name == "half_relu"
    finally:
        anyinit.unregister_activation("half_relu")


@pytest.mark.parametrize(
    "options",
    [{"mode": "vibes"}, {"gains": {"relu": 0.0}}, {"distribution": "orthogonal"}, {"seed": "zero"}],
)
def test_invalid_options_raise_before_the_model_is_touched(options):
    with pytest.raises(ConfigError):
        anyinit.initialize(object(), input_spec=(2, 2), **options)
