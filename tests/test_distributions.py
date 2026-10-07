"""Weight samplers, and the degeneracy the sinusoidal one has to avoid."""

from __future__ import annotations

import numpy as np
import pytest

from anyinit.core.distributions import DISTRIBUTIONS, sample
from anyinit.core.fan import ParamSpec


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("shape", [(128, 64), (64, 32, 3, 3)])
def test_every_distribution_has_unit_variance(distribution, shape):
    spec = ParamSpec("conv" if len(shape) > 2 else "dense", shape)
    weight = sample(spec, distribution, np.random.default_rng(0))
    assert weight.shape == shape
    assert float(weight.std()) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_sampling_is_reproducible(distribution):
    spec = ParamSpec("dense", (32, 16))
    a = sample(spec, distribution, np.random.default_rng(7))
    b = sample(spec, distribution, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_centering_zeroes_every_row_sum():
    """Check that centering zeroes every row sum.

    That is the assumption the centered form of the moment recursion rests on.
    """
    spec = ParamSpec("dense", (10, 256))
    weight = sample(spec, "normal", np.random.default_rng(0), center=True)
    assert float(np.abs(weight.sum(axis=1)).max()) < 1e-10


def test_uncentered_rows_do_not_sum_to_zero():
    spec = ParamSpec("dense", (10, 256))
    weight = sample(spec, "normal", np.random.default_rng(0), center=False)
    assert float(np.abs(weight.sum(axis=1)).max()) > 1.0


@pytest.mark.parametrize("shape", [(64, 64), (128, 64), (256, 128), (512, 512)])
def test_sinusoidal_is_near_full_rank(shape):
    """Check that a sinusoidal draw is near full rank.

    Frequencies past Nyquist alias onto each other and halve the rank; staying strictly
    inside the band costs at most two dimensions.
    """
    weight = sample(ParamSpec("dense", shape), "sinusoidal", np.random.default_rng(0))
    rank = int(np.linalg.matrix_rank(weight.astype(np.float64)))
    assert rank >= min(shape) - 2


@pytest.mark.parametrize("shape", [(64, 64), (128, 64), (256, 128), (512, 512), (1024, 256)])
def test_sinusoidal_has_no_dead_units(shape):
    weight = sample(ParamSpec("dense", shape), "sinusoidal", np.random.default_rng(0))
    assert float(weight.var(axis=1).min()) > 1e-6  # no constant rows


@pytest.mark.parametrize("shape", [(64, 64), (128, 64), (256, 128), (512, 512), (1024, 256)])
def test_sinusoidal_units_are_distinct_where_dimension_allows(shape):
    """Only asserted for shapes where distinctness is achievable.

    ``n_in`` dimensions hold at most ``n_in`` independent directions, so a layer with
    many more units than inputs *must* have correlated rows.  The sampler spreads them
    as far apart as the geometry permits rather than pretending otherwise.
    """
    weight = sample(ParamSpec("dense", shape), "sinusoidal", np.random.default_rng(0))
    normalized = weight / np.linalg.norm(weight, axis=1, keepdims=True)
    similarity = np.abs(normalized @ normalized.T)
    np.fill_diagonal(similarity, 0.0)
    assert int((similarity > 0.99).any(axis=1).sum()) == 0


def test_sinusoidal_spreads_surplus_units_as_far_as_geometry_allows():
    """Check that surplus units are spread as far as the geometry allows.

    With many more units than inputs the rows cannot be distinct, but the worst-case
    similarity should track the forced crowding rather than collapse.
    """
    weight = sample(ParamSpec("dense", (2048, 128)), "sinusoidal", np.random.default_rng(0))
    normalized = weight / np.linalg.norm(weight, axis=1, keepdims=True)
    similarity = np.abs(normalized @ normalized.T)
    np.fill_diagonal(similarity, 0.0)
    assert float(similarity.max()) < 0.999


def test_unknown_distribution_is_rejected():
    with pytest.raises(ValueError, match="unknown distribution"):
        sample(ParamSpec("dense", (4, 4)), "fractal", np.random.default_rng(0))


@pytest.mark.parametrize("fan_in", [1, 2, 4])
def test_narrow_layers_are_not_centered(fan_in):
    """Check that a narrow layer keeps its full rank.

    Centering costs the layer one input direction, which a wide layer can spare and a
    narrow one cannot: at fan_in 2 the rank would drop to one.
    """
    spec = ParamSpec("dense", (128, fan_in))
    weight = sample(spec, "normal", np.random.default_rng(0), center=True)
    rank = int(np.linalg.matrix_rank(weight.astype(np.float64)))
    assert rank == min(128, fan_in), "a narrow layer must keep its full rank"


@pytest.mark.parametrize("fan_in", [8, 32, 256])
def test_wide_layers_are_centered(fan_in):
    spec = ParamSpec("dense", (128, fan_in))
    weight = sample(spec, "normal", np.random.default_rng(0), center=True)
    assert float(np.abs(weight.sum(axis=1)).max()) < 1e-10


def test_centering_threshold_matches_can_center():
    from anyinit.core.distributions import MIN_CENTERING_FAN, can_center

    assert not can_center(ParamSpec("dense", (64, MIN_CENTERING_FAN - 1)))
    assert can_center(ParamSpec("dense", (64, MIN_CENTERING_FAN)))
    # Convolutions count their receptive field, so a 3x3 kernel over 2 channels qualifies.
    assert can_center(ParamSpec("conv", (16, 2, 3, 3)))
    assert not can_center(ParamSpec("conv", (16, 1, 2, 2)))


def test_sinusoidal_rows_sum_to_zero_by_construction():
    """Check that sinusoidal rows sum to zero without being asked.

    A sinusoid over a whole period is zero, so the layer discards its input's mean and
    the recursion has to use its centered form.
    """
    from anyinit.core.distributions import is_centered

    spec = ParamSpec("dense", (256, 256))
    weight = sample(spec, "sinusoidal", np.random.default_rng(0))
    assert float(np.abs(weight.sum(axis=1)).max()) < 1e-9
    assert is_centered("sinusoidal", spec, requested=False)


@pytest.mark.parametrize("distribution", ["normal", "uniform"])
def test_other_distributions_are_not_inherently_centered(distribution):
    from anyinit.core.distributions import is_centered

    spec = ParamSpec("dense", (256, 256))
    weight = sample(spec, distribution, np.random.default_rng(0))
    assert float(np.abs(weight.sum(axis=1)).max()) > 1.0
    assert not is_centered(distribution, spec, requested=False)
    assert is_centered(distribution, spec, requested=True)
