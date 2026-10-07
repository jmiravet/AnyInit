"""Public API surface: validation, aliases and error messages."""

from __future__ import annotations

import numpy as np
import pytest

import anyinit
from anyinit.config import InitConfig
from anyinit.errors import BackendNotFoundError, ConfigError


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("mode", "vibes", "mode must be one of"),
        ("distribution", "fractal", "distribution must be one of"),
        ("distribution", "orthogonal", "distribution must be one of"),
        ("seed", 1.5, "seed must be an int"),
        ("gains", {"relu": -1.0}, "must be positive"),
        ("gains", {"relu": "big"}, "must be a number"),
    ],
)
def test_bad_options_are_rejected(option, value, message):
    with pytest.raises(ConfigError, match=message):
        InitConfig.build(**{option: value})


def test_unknown_option_is_rejected():
    with pytest.raises(TypeError, match="unexpected keyword"):
        anyinit.initialize(object(), lr=0.1)


def test_empirical_mode_requires_data():
    with pytest.raises(ConfigError, match="needs data"):
        InitConfig.build(mode="empirical")


def test_sinusoidal_distribution_is_flagged_as_non_iid():
    config = InitConfig.build(distribution="sinusoidal")
    assert any("i.i.d." in note for note in config.notes)


@pytest.mark.parametrize(
    ("name", "params", "expected"),
    [
        ("relu", {}, 2**0.5),
        ("identity", {}, 1.0),
        ("leaky_relu", {"negative_slope": 1.0}, 1.0),
        ("relu3", None, 1.0 / 7.5 ** (1.0 / 6.0)),
    ],
)
def test_gain_lands_the_second_moment_on_one(name, params, expected):
    if params is None:
        anyinit.register_activation(lambda x: np.maximum(x, 0.0) ** 3, name=name)
        try:
            assert anyinit.gain(name) == pytest.approx(expected, rel=1e-9)
        finally:
            anyinit.unregister_activation(name)
    else:
        assert anyinit.gain(name, **params) == pytest.approx(expected, rel=1e-9)


def test_gain_of_a_bounded_activation_is_its_substituted_operating_point():
    profile = anyinit.activation_profile("tanh")
    assert profile.diagnostics.sigma_star is None
    assert 0.5 < anyinit.gain("tanh") < 3.0


def test_backend_not_found_is_actionable():
    with pytest.raises(BackendNotFoundError, match=r"Installed backends|is installed"):
        anyinit.initialize(object())


def test_activation_profile_of_a_builtin_needs_no_backend():
    assert anyinit.activation_profile("relu").chi == pytest.approx(1.0, abs=1e-6)


def test_activation_profile_rejects_an_unknown_name():
    with pytest.raises(ConfigError, match="known activations"):
        anyinit.activation_profile("telepathy")


def test_parameterised_activation_profiles_differ():
    gentle = anyinit.activation_profile("leaky_relu", negative_slope=0.01)
    steep = anyinit.activation_profile("leaky_relu", negative_slope=0.5)
    assert gentle.moments(0.0, 1.0).m2 != pytest.approx(steep.moments(0.0, 1.0).m2)


def test_register_and_unregister_round_trip():
    anyinit.register_activation(lambda x: np.maximum(x, 0.0) ** 3, name="probe3")
    try:
        assert "probe3" in anyinit.registered_activations()
        assert anyinit.activation_profile("probe3").moments(0.0, 1.0).m2 == pytest.approx(7.5)
    finally:
        anyinit.unregister_activation("probe3")
    assert "probe3" not in anyinit.registered_activations()


def test_duplicate_registration_needs_overwrite():
    anyinit.register_activation(lambda x: x, name="probe_dup")
    try:
        with pytest.raises(ValueError, match="already registered"):
            anyinit.register_activation(lambda x: x, name="probe_dup")
        anyinit.register_activation(lambda x: x * 2, name="probe_dup", overwrite=True)
    finally:
        anyinit.unregister_activation("probe_dup")


def test_numpy_callables_are_detected_automatically():
    anyinit.register_activation(lambda x: np.maximum(x, 0.0) ** 2, name="probe_np")
    try:
        # No backend needed: the profile resolves from the NumPy implementation alone.
        assert anyinit.activation_profile("probe_np").homogeneous_degree == pytest.approx(2.0)
    finally:
        anyinit.unregister_activation("probe_np")


def test_depth_error_factor_is_exported():
    assert anyinit.depth_error_factor(3.0, 4) == pytest.approx(81.0)


def test_everything_in_all_is_importable():
    for name in anyinit.__all__:
        assert hasattr(anyinit, name), name
