"""Constant factors in the IR: propagated by the recursion, and solved around."""

from __future__ import annotations

import math

import pytest

from anyinit.core import analytic
from anyinit.core.activations import BUILTIN
from anyinit.core.fan import ParamSpec
from anyinit.core.graph import ModelGraph, Node, NodeKind
from anyinit.core.moments import MomentState
from anyinit.core.profile import ActivationProfile
from anyinit.core.registry import ActivationRef
from anyinit.core.topology import build_plan

RELU = {"activation": ActivationRef("relu")}


def _solve(graph: ModelGraph):
    profiles = {
        n.id: ActivationProfile("relu", BUILTIN["relu"]) for n in graph.of_kind(NodeKind.ACTIVATION)
    }
    return analytic.solve(graph, build_plan(graph), profiles)


def test_a_layer_is_solved_for_the_factor_after_it():
    nodes = [
        Node("x", NodeKind.INPUT, "input"),
        Node("fc", NodeKind.PARAMETRIC, "Linear", ("x",), spec=ParamSpec("dense", (64, 64))),
        Node("mul", NodeKind.SCALE, "mul", ("fc",), meta={"factor": MomentState.of_values(4.0)}),
        Node("relu", NodeKind.ACTIVATION, "relu", ("mul",), meta=RELU),
        Node("out", NodeKind.OUTPUT, "output", ("relu",)),
    ]
    result = _solve(ModelGraph(nodes))

    assert result.scales["fc"] == pytest.approx(math.sqrt(2.0 / 64) / 4.0)
    assert result.states["relu"].m2 == pytest.approx(1.0)


def test_a_factor_on_a_lookup_reaches_the_next_layer():
    nodes = [
        Node("x", NodeKind.INPUT, "input"),
        Node(
            "emb", NodeKind.PARAMETRIC, "Embedding", ("x",), spec=ParamSpec("embedding", (10, 64))
        ),
        Node("mul", NodeKind.SCALE, "mul", ("emb",), meta={"factor": MomentState.of_values(8.0)}),
        Node("fc", NodeKind.PARAMETRIC, "Linear", ("mul",), spec=ParamSpec("dense", (64, 64))),
        Node("relu", NodeKind.ACTIVATION, "relu", ("fc",), meta=RELU),
        Node("out", NodeKind.OUTPUT, "output", ("relu",)),
    ]
    result = _solve(ModelGraph(nodes))

    assert result.states["mul"].m2 == pytest.approx(64.0)
    assert result.scales["fc"] == pytest.approx(math.sqrt(2.0 / 64) / 8.0)
