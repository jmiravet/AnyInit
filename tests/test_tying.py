"""Tied embedding tables, in the IR alone: which ones count, and the scale they get."""

from __future__ import annotations

import pytest

from anyinit.core import tying
from anyinit.core.fan import ParamSpec
from anyinit.core.graph import ModelGraph, Node, NodeKind
from anyinit.core.moments import MomentState

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


def test_the_report_line_gives_the_numbers_and_leaves_the_explanation_to_the_docs():
    graph = _graph(("emb", "embedding"), ("head", "dense"))
    (table,) = tying.find(graph)

    assert tying.describe(table, graph) == (
        "tied embedding (emb, head): scaled for the output layer (0.125), so the lookup "
        f"delivers E[x^2] = 0.0156; see {tying.DOCS}"
    )


def _scaled(nid: str, source: str, factor: float) -> Node:
    meta = {"factor": MomentState.of_values(factor)}
    return Node(nid, NodeKind.SCALE, "mul", (source,), meta=meta)


def _tied(*, lookup: float = 1.0, before: float = 1.0, after: float = 1.0) -> ModelGraph:
    """Lookup, an optional factor on it, then the output layer between optional factors."""
    as_lookup = {"spec": ParamSpec("embedding", (VOCAB, WIDTH)), "meta": {"tied": "t"}}
    as_output = {"spec": ParamSpec("dense", (VOCAB, WIDTH)), "meta": {"tied": "t"}}
    return ModelGraph(
        [
            Node("x", NodeKind.INPUT, "input"),
            Node("emb", NodeKind.PARAMETRIC, "Embedding", ("x",), **as_lookup),
            _scaled("in", "emb", lookup),
            Node("norm", NodeKind.NORMALIZATION, "LayerNorm", ("in",)),
            _scaled("pre", "norm", before),
            Node("head", NodeKind.PARAMETRIC, "Linear", ("pre",), **as_output),
            _scaled("post", "head", after),
            Node("out", NodeKind.OUTPUT, "output", ("post",)),
        ]
    )


@pytest.mark.parametrize(("before", "after"), [(WIDTH**-0.5, 1.0), (1.0, WIDTH**-0.5)])
def test_a_logit_factor_counts_on_either_side_of_the_output_layer(before, after):
    """PaLM scales the logits, T5 the output layer's input: both leave a table of 1."""
    (table,) = tying.find(_tied(before=before, after=after))
    assert table.scale == pytest.approx(1.0)
    assert table.lookup_m2 == pytest.approx(1.0)


def test_a_lookup_factor_is_reported_not_used():
    (table,) = tying.find(_tied(lookup=WIDTH**0.5))
    assert table.scale == pytest.approx(WIDTH**-0.5)
    assert table.lookup_m2 == pytest.approx(1.0)
