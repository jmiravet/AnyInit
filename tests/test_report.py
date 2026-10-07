"""Rendering and health checks of :class:`InitReport`, built by hand so no framework runs."""

from __future__ import annotations

import json
import math

import pytest

from anyinit.report import InitReport, LayerRecord, StabilityRecord


def _report(**overrides) -> InitReport:
    fields = {
        "backend": "pytorch",
        "mode": "analytic",
        "graph_fidelity": "graph",
        "config": {"distribution": "normal", "center": False},
        "layers": (
            LayerRecord(
                "fc1",
                "Linear",
                "relu",
                64,
                0.176777,
                predicted=1.0,
                measured=1.02,
                measured_count=4096,
                units=128,
            ),
            LayerRecord(
                "bn",
                "BatchNorm1d",
                "add→relu",
                None,
                0.5,
                is_gain=True,
                predicted=1.0,
                measured=1.6,
                measured_count=4096,
                units=128,
            ),
            LayerRecord("head", "Linear", "relu", 128, 0.125, fixed=True),
        ),
        "stability": (
            StabilityRecord("relu", 1.0, 1.4142, 1.0, (0.0, math.inf), "marginal", depth=2),
            StabilityRecord("tanh", 0.46, None, None, (0.0, 1.0), "infeasible", depth=2),
        ),
        "warnings": ("something to know",),
        "iterations": 3,
        "objective_error": 1e-9,
        "unscaled": (("pos", "a bare parameter, not a layer"),),
    }
    fields.update(overrides)
    return InitReport(**fields)


TEXT = """\
AnyInit — backend=pytorch, mode=analytic, distribution=normal, center=False, graph=graph
  solve: 3 pass(es), largest objective gap 1.00e-09

  layer  kind         activation         fan          scale    pred    meas
  ───────────────────────────────────────────────────────────────────────────
  fc1    Linear       relu                64       0.176777  1.0000  1.0200
  bn     BatchNorm1d  add→relu             -  gain=0.500000  1.0000  1.6000
  head   Linear       relu (fixed gain)  128       0.125000       -       -

Stability
  relu               chi= 1.000  sigma*=  1.4142  marginal degree=1
  tanh               chi= 0.460  sigma*=       -  infeasible
      ! tanh saturates (E[a^2] <= 1), so forward variance cannot be preserved; AnyInit \
targeted the middle of its reachable variance range instead

Validation
  largest unexplained prediction/measurement gap: 41.78% at bn  [CHECK]  (raw gap 60.00%, \
the rest is sampling noise)
  the prediction is the ensemble average and this model is one draw of it, which drifts \
further from it with depth; center=True removes most of it, mode='empirical' all of it

Not scaled
  - pos: a bare parameter, not a layer

Warnings
  - something to know"""

MARKDOWN = """\
## AnyInit — backend=pytorch, mode=analytic, distribution=normal, center=False, graph=graph
  solve: 3 pass(es), largest objective gap 1.00e-09

| layer | kind | activation | fan | scale | pred | meas |
|---|---|---|---|---|---|---|
| fc1 | Linear | relu | 64 | 0.176777 | 1.0000 | 1.0200 |
| bn | BatchNorm1d | add→relu | - | gain=0.500000 | 1.0000 | 1.6000 |
| head | Linear | relu (fixed gain) | 128 | 0.125000 | - | - |

### Stability
- relu               chi= 1.000  sigma*=  1.4142  marginal degree=1
- tanh               chi= 0.460  sigma*=       -  infeasible
  - ! tanh saturates (E[a^2] <= 1), so forward variance cannot be preserved; AnyInit \
targeted the middle of its reachable variance range instead

### Validation
- largest unexplained prediction/measurement gap: 41.78% at bn  [CHECK]  (raw gap 60.00%, \
the rest is sampling noise)
- the prediction is the ensemble average and this model is one draw of it, which drifts \
further from it with depth; center=True removes most of it, mode='empirical' all of it

### Not scaled
  - pos: a bare parameter, not a layer

### Warnings
  - something to know"""


def test_text_rendering():
    assert str(_report()) == TEXT


def test_markdown_rendering():
    assert _report().to_markdown() == MARKDOWN


def test_drift_hint_only_in_analytic_mode():
    assert "ensemble average" not in str(_report(mode="empirical"))
    centered = str(_report(config={"distribution": "normal", "center": True}))
    assert "mode='empirical' corrects it" in centered


def test_minimal_report_renders_without_optional_sections():
    text = str(_report(layers=(), stability=(), warnings=(), unscaled=()))
    for section in ("Stability", "Validation", "Not scaled", "Warnings"):
        assert section not in text


def test_to_dict_is_json_serializable():
    data = json.loads(json.dumps(_report().to_dict(), default=str))
    assert [layer["name"] for layer in data["layers"]] == ["fc1", "bn", "head"]
    assert data["unscaled"] == [{"name": "pos", "reason": "a bare parameter, not a layer"}]


def test_worst_layer_ignores_gaps_within_sampling_noise():
    report = _report()
    assert report.worst_layer is not None
    assert report.worst_layer.name == "bn"
    assert report.layers[0].excess_deviation == 0.0


def test_assert_healthy_names_every_problem():
    unstable = StabilityRecord("relu3", 3.0, None, 3.0, (0.0, math.inf), "unstable", depth=4)
    report = _report(converged=False, stability=(unstable,))
    with pytest.raises(AssertionError) as caught:
        report.assert_healthy()
    message = str(caught.value)
    assert "did not converge" in message
    assert "'bn'" in message
    assert "relu3 is not depth-stable" in message


def test_assert_healthy_passes_a_clean_report():
    clean = _report(layers=_report().layers[:1], stability=(), converged=True)
    clean.assert_healthy()


@pytest.mark.parametrize(
    ("verdict", "chi", "expected"),
    [("unstable", 3.0, "No scalar initialization is depth-stable"), ("marginal", 1.0, None)],
)
def test_stability_advice(verdict, chi, expected):
    record = StabilityRecord("act", chi, None, None, (0.0, math.inf), verdict, depth=8)
    advice = record.advice()
    if expected is None:
        assert advice is None
    else:
        assert advice is not None
        assert expected in advice
