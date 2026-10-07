"""Activation profiles: the moment map, and what gets probed automatically."""

from __future__ import annotations

import numpy as np
import pytest

from anyinit.core.activations import BUILTIN
from anyinit.core.profile import ActivationProfile

# Verified by quadrature against closed forms; see docs/reference.md.
REFERENCE = {
    "relu": (0.3989423, 0.5000000, 1.0, 1),
    "gelu": (0.2820948, 0.4252215, 1.084, None),
    "silu": (0.2066210, 0.3557755, 1.147, None),
    "elu": (0.1605206, 0.6449454, 0.898, None),
    "tanh": (0.0000000, 0.3942945, None, None),
    "sigmoid": (0.5000000, 0.2933790, None, None),
}


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_builtin_moments_match_reference(name):
    expected_mean, expected_m2, _chi, _degree = REFERENCE[name]
    state = ActivationProfile(name, BUILTIN[name]).moments(0.0, 1.0)
    assert state.mean == pytest.approx(expected_mean, abs=2e-6)
    assert state.m2 == pytest.approx(expected_m2, abs=2e-6)


@pytest.mark.parametrize("name", ["relu", "gelu", "silu", "elu"])
def test_chi_matches_reference(name):
    expected_chi = REFERENCE[name][2]
    assert ActivationProfile(name, BUILTIN[name]).chi == pytest.approx(expected_chi, abs=5e-3)


@pytest.mark.parametrize(
    ("power", "degree"),
    [(1, 1.0), (2, 2.0), (3, 3.0)],
)
def test_homogeneity_degree_is_detected(power, degree):
    profile = ActivationProfile(f"relu{power}", lambda x, p=power: np.maximum(x, 0.0) ** p)
    assert profile.homogeneous_degree == pytest.approx(degree)


@pytest.mark.parametrize("name", ["tanh", "sigmoid", "gelu", "silu", "softplus"])
def test_non_homogeneous_activations_report_none(name):
    assert ActivationProfile(name, BUILTIN[name]).homogeneous_degree is None


def test_homogeneous_shortcut_agrees_with_quadrature():
    """The closed-form scaling path must not drift from the integral it replaces."""
    relu3 = lambda x: np.maximum(x, 0.0) ** 3  # noqa: E731
    shortcut = ActivationProfile("relu3", relu3)
    explicit = ActivationProfile("relu3", relu3)
    explicit._degree_probed = True  # force the quadrature path
    for var in (0.25, 1.0, 4.0, 9.0):
        a, b = shortcut.moments(0.0, var), explicit.moments(0.0, var)
        assert a.mean == pytest.approx(b.mean, rel=1e-11)
        assert a.m2 == pytest.approx(b.m2, rel=1e-11)
        assert a.m4 == pytest.approx(b.m4, rel=1e-11)


def test_moments_are_cached():
    calls = []

    def counted(x):
        calls.append(len(x))
        return np.tanh(x)

    profile = ActivationProfile("counted", counted)
    profile.moments(0.0, 1.0)
    before = len(calls)
    profile.moments(0.0, 1.0)
    assert len(calls) == before


def test_zero_variance_evaluates_pointwise():
    state = ActivationProfile("relu", BUILTIN["relu"]).moments(2.0, 0.0)
    assert state.mean == pytest.approx(2.0)
    assert state.m2 == pytest.approx(4.0)


def test_local_slope_is_exact_for_homogeneous():
    profile = ActivationProfile("relu2", lambda x: np.maximum(x, 0.0) ** 2)
    assert profile.local_slope(0.0, 4.0) == pytest.approx(2.0)


def test_local_slope_decays_as_tanh_saturates():
    """The global chi is not usable as a step size once an activation saturates."""
    profile = ActivationProfile("tanh", BUILTIN["tanh"])
    assert profile.local_slope(0.0, 0.1) > profile.local_slope(0.0, 100.0)
    assert profile.local_slope(0.0, 100.0) < 0.1
