"""The property that matters: activation statistics hold across depth.

This is the test the v1 prototype would have failed loudly.  Its gain table drove a
20-layer SiLU network's activation second moment to 2e-07 and a sigmoid network's to
0.014; a single assertion over a range would have caught both.
"""

from __future__ import annotations

import pytest

import anyinit

from .helpers import UNBOUNDED, torch_activation_moments, torch_mlp

torch = pytest.importorskip("torch")

#: Band the activation second moment must stay inside over the whole network.
#:
#: Per mode, because the two differ in a way worth asserting.  The empirical mode measures
#: each layer and holds to a few percent.  The analytic mode propagates moments, so a
#: layer's realized draw drifts from the ensemble prediction and the drift accumulates --
#: at twenty layers it reaches about a third either side.  Both are a different world from
#: an uncorrected initialization, which arrives at 1e-07.
BANDS = {"analytic": (0.3, 2.0), "empirical": (0.5, 2.0)}


@pytest.mark.parametrize("activation", UNBOUNDED)
@pytest.mark.parametrize("mode", ["analytic", "empirical"])
def test_second_moment_holds_over_twenty_layers(activation, mode):
    torch.manual_seed(0)
    model, activation_type = torch_mlp(depth=20, activation=activation)
    anyinit.initialize(model, mode, (512, 256), seed=0)
    moments = torch_activation_moments(model, activation_type)

    low, high = BANDS[mode]
    assert len(moments) == 20
    for index, value in enumerate(moments):
        assert low < value < high, f"layer {index} of {activation} ({mode}) reached {value:.4g}"


@pytest.mark.parametrize("activation", ["tanh", "sigmoid"])
@pytest.mark.parametrize("mode", ["analytic", "empirical"])
def test_saturating_activations_stay_level_even_if_below_one(activation, mode):
    """Check that a saturating activation stays level even below one.

    It cannot hold ``E[a^2]`` at one, but the substituted target must be a fixed point of
    the depth map rather than a drifting one.
    """
    torch.manual_seed(0)
    model, activation_type = torch_mlp(depth=20, activation=activation)
    anyinit.initialize(model, mode, (512, 256), seed=0)
    moments = torch_activation_moments(model, activation_type)

    assert min(moments) > 0.05, "the signal died"
    assert max(moments) / min(moments) < 1.5, f"drifted: {moments[0]:.4g} -> {moments[-1]:.4g}"


@pytest.mark.parametrize("distribution", ["normal", "uniform"])
def test_analytic_handles_the_iid_distributions(distribution):
    torch.manual_seed(0)
    model, activation_type = torch_mlp(depth=20)
    anyinit.initialize(model, "analytic", (512, 256), seed=0, distribution=distribution)
    low, high = BANDS["analytic"]
    moments = torch_activation_moments(model, activation_type)
    assert all(low < value < high for value in moments), moments


@pytest.mark.parametrize("distribution", ["normal", "uniform", "sinusoidal"])
def test_empirical_handles_every_distribution(distribution):
    """Check that the empirical mode holds every distribution.

    Sinusoidal rows are not independent, so the analytic recursion only approximates
    them; the empirical mode measures instead.
    """
    torch.manual_seed(0)
    model, activation_type = torch_mlp(depth=20)
    anyinit.initialize(model, "empirical", (512, 256), seed=0, distribution=distribution)
    low, high = BANDS["empirical"]
    moments = torch_activation_moments(model, activation_type)
    assert all(low < value < high for value in moments), moments


def test_a_badly_scaled_network_fails_the_same_property():
    """Check that the band is not vacuous.

    The stock initialization on a deep SiLU stack must violate it.
    """
    torch.manual_seed(0)
    model, activation_type = torch_mlp(depth=20, activation="silu")
    moments = torch_activation_moments(model, activation_type)
    low, high = BANDS["analytic"]
    assert not all(low < value < high for value in moments)
