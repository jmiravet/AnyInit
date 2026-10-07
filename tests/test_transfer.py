"""Moment transfer functions, against values worked out by hand."""

from __future__ import annotations

import math

import pytest

from anyinit.core.activations import BUILTIN
from anyinit.core.fan import ParamSpec
from anyinit.core.moments import MomentState
from anyinit.core.profile import ActivationProfile
from anyinit.core.transfer import (
    through_activation,
    through_average_pool,
    through_dropout,
    through_embedding,
    through_linear,
    through_merge,
    through_normalization,
)

UNIT = MomentState.standard_normal()


def test_linear_scales_by_fan_times_variance():
    spec = ParamSpec("dense", (64, 100))
    state = through_linear(UNIT, spec, sigma_w=0.1)
    assert state.m2 == pytest.approx(100 * 0.01 * 1.0)


def test_linear_uses_the_second_moment_not_the_variance():
    """A nonzero activation mean feeds its square into the next layer's variance."""
    offset = MomentState(mean=2.0, m2=5.0, m4=75.0)
    spec = ParamSpec("dense", (8, 10))
    assert through_linear(offset, spec, sigma_w=0.1).m2 == pytest.approx(10 * 0.01 * 5.0)


def test_centered_linear_uses_the_variance():
    """Centered weights annihilate the DC component instead of propagating it."""
    offset = MomentState(mean=2.0, m2=5.0, m4=75.0)
    spec = ParamSpec("dense", (8, 10))
    state = through_linear(offset, spec, sigma_w=0.1, centered=True)
    assert state.m2 == pytest.approx(10 * 0.01 * offset.var)


def test_kaiming_scale_gives_unit_second_moment_after_relu():
    """The classic result, reproduced by the general machinery."""
    spec = ParamSpec("dense", (256, 256))
    sigma = math.sqrt(2.0 / 256.0)
    pre = through_linear(UNIT, spec, sigma_w=sigma)
    post = through_activation(pre, ActivationProfile("relu", BUILTIN["relu"]))
    assert post.m2 == pytest.approx(1.0, abs=1e-9)


def test_embedding_ignores_fan():
    """A lookup returns a row, so its moments are the table's own."""
    assert through_embedding(0.5).m2 == pytest.approx(0.25)


def test_normalization_discards_the_incoming_scale():
    """Check that normalization discards the incoming scale.

    Whatever arrives, the output is unit variance times gamma, so the recursion restarts
    here and the gain is the knob that matters.
    """
    assert through_normalization(1.0).m2 == pytest.approx(1.0)
    assert through_normalization(2.0).m2 == pytest.approx(4.0)
    assert through_normalization(2.0, beta=1.0).mean == pytest.approx(1.0)
    assert through_normalization(2.0, beta=1.0).m2 == pytest.approx(5.0)


def test_add_sums_second_moments():
    a = MomentState(0.0, 1.0, 3.0)
    assert through_merge([a, a], "add").m2 == pytest.approx(2.0)


def test_add_includes_the_cross_term_for_nonzero_means():
    a = MomentState(1.0, 2.0, 12.0)
    merged = through_merge([a, a], "add")
    assert merged.mean == pytest.approx(2.0)
    assert merged.m2 == pytest.approx(2.0 + 2.0 + 2 * 1.0 * 1.0)


def test_multiply_multiplies_moments():
    a = MomentState(0.0, 2.0, 12.0)
    b = MomentState(0.0, 3.0, 27.0)
    assert through_merge([a, b], "mul").m2 == pytest.approx(6.0)


def test_concat_averages_by_element_count():
    a = MomentState(0.0, 1.0, 3.0)
    b = MomentState(0.0, 3.0, 27.0)
    assert through_merge([a, b], "cat").m2 == pytest.approx(2.0)


def test_single_branch_merge_is_the_identity():
    a = MomentState(0.5, 2.0, 12.0)
    assert through_merge([a], "add") == a


def test_dropout_inflates_variance_at_train_time():
    """Check that dropout inflates the variance at train time.

    Inverted dropout scales survivors by ``1/(1-p)``.
    """
    assert through_dropout(UNIT, 0.5).m2 == pytest.approx(2.0)
    assert through_dropout(UNIT, 0.0).m2 == pytest.approx(1.0)


def test_average_pool_reduces_variance():
    pooled = through_average_pool(UNIT, window=4)
    assert pooled.m2 < UNIT.m2
    assert through_average_pool(UNIT, window=1) == UNIT


def test_heavy_tailed_input_triggers_the_mixture_correction():
    """Check that a heavy-tailed input triggers the mixture correction.

    The pre-activation is then not Gaussian, and a high-order activation's high moments
    come out too small if that is ignored.
    """
    profile = ActivationProfile("relu3", lambda x: __import__("numpy").maximum(x, 0.0) ** 3)
    heavy = MomentState(mean=0.0, m2=1.0, m4=90.0)  # kurtosis 90, far from Gaussian
    spec_fan, sigma = 256.0, 1.0 / 16.0
    plain = through_activation(heavy, profile)  # no layer given: plain Gaussian
    corrected = through_activation(heavy, profile, fan=spec_fan, sigma_w=sigma)
    assert corrected.m2 > plain.m2
