"""Tied embedding tables, in the IR alone: which ones count, and the scale they get."""

from __future__ import annotations

import pytest

from anyinit.core import tying
from anyinit.core.fan import ParamSpec
from anyinit.core.graph import ModelGraph, Node, NodeKind

VOCAB, WIDTH = 1000, 64


def _graph(*readers: tuple[str, str]) -> ModelGraph:
    """Input, then one node per ``(id, kind)``, all reading the table ``t``."""
    nodes = [Node("x", NodeKind.INPUT, "input")]
    for nid, kind in readers:
        nodes.append(
            Node(
                nid,
                NodeKind.PARAMETRIC,
                kind,
                (nodes[-1].id,),
                spec=ParamSpec(kind, (VOCAB, WIDTH)),
                meta={"tied": "t", "path": nid},
            )
        )
    nodes.append(Node("out", NodeKind.OUTPUT, "output", (nodes[-1].id,)))
    return ModelGraph(nodes)


def test_a_table_read_in_both_roles_takes_the_output_scale():
    (table,) = tying.find(_graph(("emb", "embedding"), ("head", "dense")))

    assert table.lookups == ("emb",)
    assert table.outputs == ("head",)
    assert table.scale == pytest.approx(WIDTH**-0.5)
    assert tying.scales([table]) == {"emb": WIDTH**-0.5, "head": WIDTH**-0.5}


def test_a_table_read_in_one_role_only_is_not_tied():
    assert tying.find(_graph(("a", "embedding"), ("b", "embedding"))) == ()
    assert tying.find(_graph(("a", "dense"), ("b", "dense"))) == ()


def test_unmarked_layers_are_never_tied():
    nodes = [
        Node("x", NodeKind.INPUT, "input"),
        Node("emb", NodeKind.PARAMETRIC, "E", ("x",), spec=ParamSpec("embedding", (10, 4))),
        Node("head", NodeKind.PARAMETRIC, "L", ("emb",), spec=ParamSpec("dense", (10, 4))),
    ]
    assert tying.find(ModelGraph(nodes)) == ()


def test_the_report_line_names_both_roles_and_points_to_the_measurements():
    graph = _graph(("emb", "embedding"), ("head", "dense"))
    (table,) = tying.find(graph)
    line = tying.describe(table, graph)

    assert "emb is also the output layer head" in line
    assert f"1/sqrt({WIDTH})" in line
    assert tying.DOCS in line
