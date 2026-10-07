"""Shared fixtures.

Every backend test is skipped rather than failed when its framework is absent, so the
suite passes with any subset of PyTorch, Keras and JAX installed.  The core tests need
nothing but NumPy.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

# Keras 3 defaults to the TensorFlow backend.  When TensorFlow is absent but another
# supported backend is present, point Keras at that instead so the Keras adapter is
# actually exercised -- it talks to the Keras API, which is identical either way.
if "KERAS_BACKEND" not in os.environ and importlib.util.find_spec("tensorflow") is None:
    for _candidate in ("torch", "jax"):
        if importlib.util.find_spec(_candidate) is not None:
            os.environ["KERAS_BACKEND"] = _candidate
            break

from anyinit.core.fan import ParamSpec
from anyinit.core.graph import ModelGraph, Node, NodeKind
from anyinit.core.registry import ActivationRef


@pytest.fixture
def mlp_graph():
    """Factory for a straight-line MLP in the IR, with no framework involved."""

    def build(depth: int = 4, width: int = 64, activation: str = "relu", in_features: int = 32):
        nodes = [Node("x", NodeKind.INPUT, "input")]
        previous = "x"
        for i in range(depth):
            fan_in = in_features if i == 0 else width
            nodes.append(
                Node(
                    f"fc{i}",
                    NodeKind.PARAMETRIC,
                    "Linear",
                    (previous,),
                    spec=ParamSpec("dense", (width, fan_in)),
                    handle=f"fc{i}",
                )
            )
            nodes.append(
                Node(
                    f"act{i}",
                    NodeKind.ACTIVATION,
                    activation,
                    (f"fc{i}",),
                    meta={"activation": ActivationRef(activation)},
                )
            )
            previous = f"act{i}"
        nodes.append(Node("out", NodeKind.OUTPUT, "output", (previous,)))
        return ModelGraph(nodes)

    return build


@pytest.fixture
def residual_graph():
    r"""Two branches merging into one activation.

    fc_main -> bn_main ---\\
                               add -> relu
    fc_skip -> bn_skip ---/
    """
    nodes = [
        Node("x", NodeKind.INPUT, "input"),
        Node("fc_main", NodeKind.PARAMETRIC, "Linear", ("x",), spec=ParamSpec("dense", (64, 64))),
        Node(
            "bn_main",
            NodeKind.NORMALIZATION,
            "BatchNorm",
            ("fc_main",),
            spec=ParamSpec("norm", (64,)),
            handle="bn_main",
        ),
        Node("fc_skip", NodeKind.PARAMETRIC, "Linear", ("x",), spec=ParamSpec("dense", (64, 64))),
        Node(
            "bn_skip",
            NodeKind.NORMALIZATION,
            "BatchNorm",
            ("fc_skip",),
            spec=ParamSpec("norm", (64,)),
            handle="bn_skip",
        ),
        Node("add", NodeKind.MERGE, "add", ("bn_main", "bn_skip")),
        Node(
            "relu",
            NodeKind.ACTIVATION,
            "relu",
            ("add",),
            meta={"activation": ActivationRef("relu")},
        ),
        Node("out", NodeKind.OUTPUT, "output", ("relu",)),
    ]
    return ModelGraph(nodes)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running training regressions")
