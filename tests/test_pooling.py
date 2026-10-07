"""Pooling, which decides where an objective has to be enforced.

Max pooling multiplies the second moment substantially — about five for a 3x3 window —
so an objective placed at the activation leaves the next layer reading five times the
target. In a residual network that is unrecoverable: the identity skip carries the
inflated level into a block whose only knob is a normalization gain inside it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import anyinit
from anyinit.core.activations import BUILTIN
from anyinit.core.graph import ModelGraph, Node, NodeKind
from anyinit.core.profile import ActivationProfile
from anyinit.core.quadrature import max_of_normals
from anyinit.core.registry import ActivationRef
from anyinit.core.topology import build_plan, objective_node
from anyinit.core.transfer import through_max_pool

# E[max of 2 standard normals] = 1/sqrt(pi), and E[max^2] = 1 exactly.
TWO_MAX_MEAN = 1.0 / math.sqrt(math.pi)


def test_max_of_two_normals_matches_closed_form():
    mean, second = max_of_normals(0.0, 1.0, 2.0)
    assert mean == pytest.approx(TWO_MAX_MEAN, abs=1e-9)
    assert second == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("count", [2, 3, 4, 9, 16])
def test_max_of_normals_matches_monte_carlo(count):
    quad_mean, quad_second = max_of_normals(0.0, 1.0, float(count))
    sample = np.random.default_rng(0).standard_normal((300_000, count)).max(axis=1)
    assert quad_mean == pytest.approx(float(sample.mean()), abs=0.01)
    assert quad_second == pytest.approx(float((sample**2).mean()), abs=0.02)


def test_max_pooling_of_a_relu_output_uses_the_pre_activation():
    """Check that pooling after a rectifier integrates the pre-activation maximum.

    ``max_i relu(z_i) == relu(max_i z_i)``, and fitting a Gaussian to the rectifier's own
    moments understates the result because half its mass sits at zero.
    """
    profile = ActivationProfile("relu", BUILTIN["relu"])
    exact = profile.max_moments(0.0, 2.0, 9.0)

    # relu(max of 9 N(0, 2)) equals sqrt(2) * (max of 9 standard normals) except on the
    # 2**-9 of draws where all nine are negative and the rectifier clamps to zero, so the
    # true value sits just below.
    _mean, second = max_of_normals(0.0, 1.0, 9.0)
    unclamped = 2.0 * second
    assert exact.m2 < unclamped
    assert exact.m2 == pytest.approx(unclamped, rel=1e-3)

    # Fitting a Gaussian to the ReLU's own mean and variance understates the pooled
    # second moment by about a third: 3.45 against 5.13.
    post_activation = profile.moments(0.0, 2.0)
    fitted = through_max_pool(post_activation, 9)
    assert fitted.m2 < exact.m2 * 0.75


def test_pooling_scalar_kernel_is_raised_to_the_spatial_rank():
    """Check that a scalar kernel size is raised to the spatial rank.

    MaxPool2d(3) pools nine values, not three.
    """
    torch = pytest.importorskip("torch")
    nn = torch.nn
    from anyinit.backends import resolve

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        def forward(self, x):
            return self.pool(torch.relu(self.conv(x)))

    graph = resolve(Net()).build_graph(Net())
    pool = next(n for n in graph.of_kind(NodeKind.POOL))
    assert pool.meta["window"] == 9


def test_objective_moves_past_the_pooling():
    nodes = [
        Node("x", NodeKind.INPUT, "input"),
        Node("fc", NodeKind.PARAMETRIC, "Linear", ("x",)),
        Node(
            "relu", NodeKind.ACTIVATION, "relu", ("fc",), meta={"activation": ActivationRef("relu")}
        ),
        Node("pool", NodeKind.POOL, "MaxPool2d", ("relu",), meta={"pool": "max", "window": 9}),
        Node("fc2", NodeKind.PARAMETRIC, "Linear", ("pool",)),
        Node("out", NodeKind.OUTPUT, "output", ("fc2",)),
    ]
    graph = ModelGraph(nodes)
    assert objective_node(graph, "relu") == "pool"

    plan = build_plan(graph)
    assert plan.activation["fc"] == "relu"
    assert plan.measurement["fc"] == "pool"


def test_objective_stops_at_a_branch():
    """A node feeding two places is ambiguous, so the search does not move past it."""
    nodes = [
        Node("x", NodeKind.INPUT, "input"),
        Node("fc", NodeKind.PARAMETRIC, "Linear", ("x",)),
        Node(
            "relu", NodeKind.ACTIVATION, "relu", ("fc",), meta={"activation": ActivationRef("relu")}
        ),
        Node("a", NodeKind.SHAPE, "flatten", ("relu",)),
        Node("b", NodeKind.SHAPE, "flatten", ("relu",)),
        Node("merge", NodeKind.MERGE, "add", ("a", "b")),
        Node("out", NodeKind.OUTPUT, "output", ("merge",)),
    ]
    assert objective_node(ModelGraph(nodes), "relu") == "relu"


def test_stem_with_pooling_delivers_the_target_to_the_next_layer():
    """Check that a stem with pooling delivers the target to the next layer.

    ``conv -> norm -> relu -> maxpool`` must leave the pooled output at the target rather
    than the activation.
    """
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Stem(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 7, stride=2, padding=3, bias=False)
            self.norm = nn.BatchNorm2d(16)
            self.pool = nn.MaxPool2d(3, stride=2, padding=1)
            self.head = nn.Conv2d(16, 16, 3, padding=1, bias=False)

        def forward(self, x):
            return self.head(self.pool(torch.relu(self.norm(self.conv(x)))))

    torch.manual_seed(0)
    model = Stem()
    report = anyinit.initialize(model, input_spec=(64, 3, 32, 32), seed=0)
    assert report.converged

    model.train()
    captured = {}
    handle = model.pool.register_forward_hook(
        lambda _m, _i, out: captured.update(m2=float(out.pow(2).mean()))
    )
    with torch.no_grad():
        model(torch.randn(256, 3, 32, 32))
    handle.remove()
    assert captured["m2"] == pytest.approx(1.0, rel=0.25)


@pytest.mark.parametrize("name", ["resnet18", "resnet34"])
def test_resnets_converge(name):
    """Residual stacks couple branches, so the solve has to settle across them."""
    torchvision = pytest.importorskip("torchvision")
    import torch

    torch.manual_seed(0)
    model = getattr(torchvision.models, name)(weights=None)
    report = anyinit.initialize(model, input_spec=(32, 3, 32, 32), seed=0)
    assert report.converged, report.warnings
    assert report.objective_error < 5e-3
