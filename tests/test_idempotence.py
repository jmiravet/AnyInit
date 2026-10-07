"""Initializing twice must equal initializing once.

The v1 prototype multiplied normalization gains instead of assigning them, so gamma
walked 1.414 -> 2.000 -> 2.828 across three calls while the weights, which were
normalized, stayed put.  Two different behaviors in one function.
"""

from __future__ import annotations

import pytest

import anyinit

torch = pytest.importorskip("torch")
nn = torch.nn


def model_with_norm():
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def snapshot(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def assert_same(a, b):
    assert a.keys() == b.keys()
    for name in a:
        assert torch.allclose(a[name], b[name], atol=1e-12), name


@pytest.mark.parametrize("mode", ["analytic", "empirical"])
def test_repeated_initialization_is_a_fixed_point(mode):
    torch.manual_seed(0)
    model = model_with_norm()
    anyinit.initialize(model, mode, (128, 32), seed=0)
    once = snapshot(model)
    anyinit.initialize(model, mode, (128, 32), seed=0)
    twice = snapshot(model)
    anyinit.initialize(model, mode, (128, 32), seed=0)
    assert_same(once, twice)
    assert_same(once, snapshot(model))


def test_normalization_gain_is_assigned_not_multiplied():
    torch.manual_seed(0)
    model = model_with_norm()
    gains = []
    for _ in range(3):
        anyinit.initialize(model, input_spec=(128, 32), seed=0)
        gains.append(float(model[1].weight[0].detach()))
    assert gains[0] == pytest.approx(gains[1]) == pytest.approx(gains[2])


def test_same_seed_gives_identical_weights():
    results = []
    for _ in range(2):
        torch.manual_seed(999)  # deliberately different from the AnyInit seed
        model = model_with_norm()
        anyinit.initialize(model, input_spec=(128, 32), seed=4)
        results.append(snapshot(model))
    assert_same(results[0], results[1])


def test_different_seeds_give_different_weights():
    snapshots = []
    for seed in (1, 2):
        torch.manual_seed(0)
        model = model_with_norm()
        anyinit.initialize(model, input_spec=(128, 32), seed=seed)
        snapshots.append(snapshot(model))
    assert not torch.allclose(snapshots[0]["0.weight"], snapshots[1]["0.weight"])


@pytest.mark.parametrize("mode", ["analytic", "empirical"])
def test_fixed_gain_sets_the_layers_feeding_that_activation(mode):
    """A fixed gain bypasses the solve for its activation and leaves the others to it."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.Tanh())
    report = anyinit.initialize(model, mode, torch.randn(128, 32), seed=0, gains={"relu": 2.0})
    assert float(model[0].weight.std()) == pytest.approx(2.0 / 32**0.5, rel=1e-3)
    assert [record.fixed for record in report.layers] == [True, False]
    assert report.converged


def test_a_gain_for_an_absent_activation_is_reported():
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
    report = anyinit.initialize(model, seed=0, gains={"rlu": 1.0})
    assert any("'rlu'" in warning for warning in report.warnings)
