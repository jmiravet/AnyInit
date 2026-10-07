"""The two modes must agree.

They share no code below the plan: one integrates a Gaussian, the other measures a
batch.  Where their assumptions hold they should land on the same scales, which makes
this a check on both implementations at once -- it is unlikely they would be wrong in
the same direction.
"""

from __future__ import annotations

import pytest

import anyinit

from .helpers import UNBOUNDED, torch_mlp

torch = pytest.importorskip("torch")


def scales(mode: str, activation: str, depth: int = 6, seed: int = 0):
    torch.manual_seed(0)
    model, _ = torch_mlp(depth=depth, width=256, activation=activation)
    report = anyinit.initialize(model, mode, (4096, 256), seed=seed)
    return [r.scale for r in report.layers if r.scale is not None]


@pytest.mark.parametrize("activation", UNBOUNDED)
def test_modes_agree_within_a_few_percent(activation):
    analytic = scales("analytic", activation)
    empirical = scales("empirical", activation)
    assert len(analytic) == len(empirical)
    for index, (a, e) in enumerate(zip(analytic, empirical, strict=False)):
        assert e == pytest.approx(a, rel=0.08), (
            f"layer {index} of {activation}: analytic {a:.6f} vs empirical {e:.6f}"
        )


@pytest.mark.parametrize("activation", ["tanh", "sigmoid"])
def test_modes_agree_on_the_substituted_target_too(activation):
    """Check that both modes agree on the substituted target.

    The substitution is a policy decision, so switching mode must not change it.
    """
    analytic = scales("analytic", activation, depth=4)
    empirical = scales("empirical", activation, depth=4)
    for a, e in zip(analytic, empirical, strict=False):
        assert e == pytest.approx(a, rel=0.15)


def test_analytic_validation_agrees_with_measurement():
    torch.manual_seed(0)
    model, _ = torch_mlp(depth=8, width=256, activation="relu")
    report = anyinit.initialize(model, "analytic", (1024, 256), seed=0)
    report.assert_healthy(tol=0.15)
    assert report.max_deviation is not None
    assert report.max_deviation < 0.15
