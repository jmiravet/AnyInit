"""Gaussian integration against values known in closed form."""

from __future__ import annotations

import math

import numpy as np
import pytest

from anyinit.core.quadrature import (
    detect_kinks,
    gamma_laguerre_nodes,
    gauss_legendre_nodes,
    integrate,
    scale_mixture_moments,
)

RELU = lambda x: np.maximum(x, 0.0)  # noqa: E731
RELU3 = lambda x: np.maximum(x, 0.0) ** 3  # noqa: E731

# E[relu(x)^3] = E[|x|^3]/2 = sqrt(2/pi);  E[relu(x)^6] = E[x^6]/2 = 15/2
RELU3_M1 = math.sqrt(2.0 / math.pi)
RELU3_M2 = 7.5


def moments(f, mu=0.0, sigma=1.0, powers=(1, 2), **kwargs):
    kinks = detect_kinks(f, mu - 10.0 * sigma, mu + 10.0 * sigma)
    z, w = gauss_legendre_nodes(mu, sigma, kinks=kinks, **kwargs)
    return integrate(f(z), w, powers)


def test_weights_sum_to_one():
    _z, w = gauss_legendre_nodes(0.0, 1.0)
    assert w.sum() == pytest.approx(1.0, abs=1e-15)


def test_relu_cubed_is_exact():
    m1, m2 = moments(RELU3)
    assert m1 == pytest.approx(RELU3_M1, abs=1e-12)
    assert m2 == pytest.approx(RELU3_M2, abs=1e-11)


@pytest.mark.parametrize("panels", [7, 8, 13, 32, 64])
def test_exact_at_any_panel_count(panels):
    """Accuracy must not depend on a kink happening to land on a panel edge."""
    m1, m2 = moments(RELU3, panels=panels)
    assert m1 == pytest.approx(RELU3_M1, abs=1e-11)
    assert m2 == pytest.approx(RELU3_M2, abs=1e-10)


def test_relu_gain_is_root_two():
    _m1, m2 = moments(RELU)
    assert 1.0 / math.sqrt(m2) == pytest.approx(math.sqrt(2.0), abs=1e-12)


def test_moments_of_shifted_gaussian():
    """E[z] and E[z^2] for a non-standard normal, as a basic sanity check."""
    m1, m2 = moments(lambda x: x, mu=1.5, sigma=2.0)
    assert m1 == pytest.approx(1.5, abs=1e-12)
    assert m2 == pytest.approx(1.5**2 + 2.0**2, abs=1e-11)


def test_detect_kinks_finds_the_origin_and_the_clip():
    kinks = detect_kinks(lambda x: np.clip(x, 0.0, 6.0), -10.0, 10.0)
    assert any(abs(k) < 1e-6 for k in kinks)
    assert any(abs(k - 6.0) < 1e-6 for k in kinks)


def test_detect_kinks_on_a_smooth_function_adds_only_the_origin():
    assert detect_kinks(np.tanh, -10.0, 10.0) == [0.0]


def test_integrate_rejects_a_non_elementwise_result():
    _z, w = gauss_legendre_nodes(0.0, 1.0)
    with pytest.raises(ValueError, match="elementwise"):
        integrate(np.ones(3), w, (2,))


def test_integrate_rejects_non_finite_values():
    z, w = gauss_legendre_nodes(0.0, 1.0)
    with pytest.raises(ValueError, match="non-finite"):
        integrate(np.full_like(z, np.inf), w, (2,))


def test_gamma_quadrature_reproduces_its_own_moments():
    mean, var = 2.0, 0.75
    scales, weights = gamma_laguerre_nodes(mean, var)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-12)
    assert float((weights * scales).sum()) == pytest.approx(mean, rel=1e-6)
    second = float((weights * scales**2).sum())
    assert second - mean**2 == pytest.approx(var, rel=1e-5)


def test_scale_mixture_matches_plain_quadrature_with_no_dispersion():
    plain = moments(RELU3, sigma=math.sqrt(2.0), powers=(1, 2, 4))
    mixed = scale_mixture_moments(RELU3, 2.0, 0.0, (1, 2, 4), kinks=[0.0])
    assert mixed == pytest.approx(plain, rel=1e-9)


def test_scale_mixture_raises_the_high_moments():
    """Dispersion in the mixing scale inflates high moments; that is the whole point."""
    plain = scale_mixture_moments(RELU3, 1.0, 0.0, (2,), kinks=[0.0])[0]
    mixed = scale_mixture_moments(RELU3, 1.0, 0.5, (2,), kinks=[0.0])[0]
    assert mixed > plain * 1.5
