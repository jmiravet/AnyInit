"""The IR's own invariants."""

from __future__ import annotations

import pytest

from anyinit.core.graph import FIDELITY_GRAPH, ModelGraph, Node, NodeKind


def test_edges_and_order(mlp_graph):
    graph = mlp_graph(depth=2)
    assert len(graph) == 6
    assert graph.predecessors("act0") == ("fc0",)
    assert graph.successors("fc0") == ("act0",)
    assert graph.fidelity == FIDELITY_GRAPH


def test_of_kind_preserves_topological_order(mlp_graph):
    graph = mlp_graph(depth=3)
    assert [n.id for n in graph.of_kind(NodeKind.ACTIVATION)] == ["act0", "act1", "act2"]
    assert [n.id for n in graph.scalable] == ["fc0", "fc1", "fc2"]


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate node id"):
        ModelGraph([Node("a", NodeKind.INPUT, "i"), Node("a", NodeKind.OUTPUT, "o")])


def test_unknown_input_is_rejected():
    with pytest.raises(ValueError, match="unknown input"):
        ModelGraph([Node("a", NodeKind.OUTPUT, "o", ("ghost",))])


def test_out_of_order_nodes_are_rejected():
    """Every consumer assumes topological order, so it is checked once at construction."""
    nodes = [
        Node("consumer", NodeKind.OUTPUT, "o", ("producer",)),
        Node("producer", NodeKind.INPUT, "i"),
    ]
    with pytest.raises(ValueError, match="topological order"):
        ModelGraph(nodes)


def test_replace_cannot_rewire(mlp_graph):
    graph = mlp_graph(depth=1)
    with pytest.raises(ValueError, match="edges"):
        graph.replace(Node("act0", NodeKind.ACTIVATION, "relu", ()))


def test_replace_updates_metadata(mlp_graph):
    graph = mlp_graph(depth=1)
    graph.replace(graph["act0"].with_meta(flavor="spicy"))
    assert graph["act0"].meta["flavor"] == "spicy"
