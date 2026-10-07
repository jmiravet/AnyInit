"""Depth-stability diagnosis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from anyinit.core.activations import BUILTIN
from anyinit.core.profile import ActivationProfile
from anyinit.core.stability import (
    DRIFT_LIMIT,
    classify,
    depth_error_factor,
    diagnose,
    lyapunov_slope,
    stable_depth,
    substituted_sigma,
)


def power(p):
    return ActivationProfile(f"relu{p}", lambda x, q=p: np.maximum(x, 0.0) ** q)


@pytest.mark.parametrize(("p", "chi"), [(1, 1.0), (2, 2.0), (3, 3.0)])
def test_chi_equals_homogeneity_degree(p, chi):
    """For a positively homogeneous activation the two coincide exactly."""
    assert power(p).chi == pytest.approx(chi, abs=1e-6)


def test_relu_sits_at_the_edge_of_chaos():
    diag = diagnose(ActivationProfile("relu", BUILTIN["relu"]))
    assert diag.verdict == "marginal"
    assert diag.sigma_star == pytest.approx(math.sqrt(2.0), abs=1e-9)


@pytest.mark.parametrize(("p", "sigma_star"), [(2, 0.9036), (3, 0.7148)])
def test_sigma_star_for_powers(p, sigma_star):
    assert power(p).diagnostics.sigma_star == pytest.approx(sigma_star, abs=1e-4)


@pytest.mark.parametrize("p", [2, 3])
def test_powers_above_one_are_expansive_without_a_depth(p):
    assert power(p).diagnostics.verdict == "expansive"


@pytest.mark.parametrize(
    ("chi", "depth", "verdict"),
    [
        (2.0, 3, "expansive"),
        (2.0, 4, "unstable"),
        (1.147, 16, "expansive"),
        (1.147, 17, "unstable"),
    ],
)
def test_expansive_turns_unstable_with_depth(chi, depth, verdict):
    """An error growing chi-fold per layer is tolerable until chi**depth passes the limit."""
    assert classify(chi, 1.0, depth) == verdict


def test_stable_depth_is_where_the_drift_limit_is_reached():
    assert stable_depth(1.0) == math.inf
    assert DRIFT_LIMIT ** (1.0 / stable_depth(2.0)) == pytest.approx(2.0)


@pytest.mark.parametrize("name", ["tanh", "sigmoid"])
def test_saturating_chi_is_read_at_the_substituted_operating_point(name):
    profile = ActivationProfile(name, BUILTIN[name])
    sigma = substituted_sigma(profile)
    assert sigma is not None
    low, high = profile.diagnostics.feasible_var
    assert profile.moments(0.0, sigma * sigma).var == pytest.approx(0.5 * (low + high), rel=1e-6)
    assert profile.diagnostics.chi == pytest.approx(
        lyapunov_slope(profile, sigma, metric="var"), rel=1e-9
    )


@pytest.mark.parametrize("name", ["tanh", "sigmoid", "hardsigmoid", "softsign"])
def test_saturating_activations_are_infeasible(name):
    """Check that a bounded activation reports its target as unreachable.

    It must say so rather than returning the edge of its own search bracket.
    """
    diag = diagnose(ActivationProfile(name, BUILTIN[name]))
    assert diag.verdict == "infeasible"
    assert diag.sigma_star is None
    assert math.isfinite(diag.feasible_m2[1])


def test_selu_preserves_unit_variance_by_construction():
    """Check that SELU comes out variance-preserving at unit variance.

    That is its design property, so agreeing with it exercises the whole pipeline.
    """
    diag = diagnose(ActivationProfile("selu", BUILTIN["selu"]))
    assert diag.sigma_star == pytest.approx(1.0, abs=2e-3)


@pytest.mark.parametrize("name", ["relu", "gelu", "silu", "elu", "mish", "softplus"])
def test_unbounded_activations_are_feasible(name):
    diag = diagnose(ActivationProfile(name, BUILTIN[name]))
    assert diag.sigma_star is not None
    assert diag.feasible_m2[1] == math.inf


def test_sigmoid_second_moment_floor_is_its_mean_squared():
    """sigmoid(0) = 1/2, so E[a^2] can never drop below 1/4."""
    low, high = ActivationProfile("sigmoid", BUILTIN["sigmoid"]).feasible_m2
    assert low == pytest.approx(0.25, abs=1e-3)
    assert high == pytest.approx(0.5, abs=1e-2)


def test_classify_bands():
    assert classify(0.5, 1.0) == "contractive"
    assert classify(1.0, 1.0) == "marginal"
    assert classify(2.0, 1.0) == "expansive"
    assert classify(2.0, 1.0, depth=20) == "unstable"
    assert classify(1.0, None) == "infeasible"


def test_depth_error_factor_compounds():
    assert depth_error_factor(3.0, 20) == pytest.approx(3.0**20)
    assert depth_error_factor(1.0, 1000) == pytest.approx(1.0)
