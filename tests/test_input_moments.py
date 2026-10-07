"""The boundary condition of the analytic recursion.

Assuming a standard normal input is right when AnyInit synthesizes the batch and wrong
when the caller supplies one that is not: data uniform on [-2, 2] has E[x^2] = 4/3, and
assuming otherwise scales every layer in the network by that factor.
"""

from __future__ import annotations

import pytest

import anyinit

torch = pytest.importorskip("torch")
nn = torch.nn


def mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1)
    )


def activation_levels(model: nn.Module, batch: torch.Tensor) -> list[float]:
    hidden = batch
    levels = []
    with torch.no_grad():
        for layer in model:
            hidden = layer(hidden)
            if isinstance(layer, nn.SiLU):
                levels.append(float(hidden.pow(2).mean()))
    return levels


def test_moments_are_read_from_a_supplied_batch():
    torch.manual_seed(0)
    model = mlp()
    batch = torch.rand(4096, 2) * 4 - 2  # E[x^2] = 4/3, not 1
    report = anyinit.initialize(model, input_spec=batch, seed=0)

    for level in activation_levels(model, batch):
        assert level == pytest.approx(1.0, rel=0.15)
    assert any("batch's own moments" in w for w in report.warnings)


def test_a_shape_means_standard_normal_is_assumed():
    torch.manual_seed(0)
    model = mlp()
    report = anyinit.initialize(model, input_spec=(256, 2), seed=0)
    assert not any("batch's own moments" in w for w in report.warnings)

    for level in activation_levels(model, torch.randn(4096, 2)):
        assert level == pytest.approx(1.0, rel=0.15)


def test_assuming_normal_over_scaled_data_is_visibly_wrong():
    """The contrast that makes the first test meaningful."""
    torch.manual_seed(0)
    model = mlp()
    anyinit.initialize(model, input_spec=(256, 2), seed=0)

    batch = torch.rand(4096, 2) * 4 - 2
    first = activation_levels(model, batch)[0]
    assert first > 1.2, "a 4/3 input second moment should show up as a 4/3 activation level"


def test_callable_input_spec_is_sampled_for_its_moments():
    torch.manual_seed(0)
    model = mlp()
    report = anyinit.initialize(
        model,
        "analytic",
        lambda: torch.rand(1024, 2) * 4 - 2,
        seed=0,
    )
    assert any("batch's own moments" in w for w in report.warnings)
