"""Embedding tables that double as the output layer.

With tied weights one table is read twice.  As a lookup nothing sums over it, so its rows
enter the network at the table's own scale and want a scale of 1.  As the output layer it
sums over ``d_model`` inputs, so it wants ``1/sqrt(d_model)`` for unit-variance logits.
No single scale serves both.

The output role decides.  At the lookup's scale the logits' standard deviation is
``sqrt(d_model)`` and the initial loss grows with width; at the output's scale the lookup
merely enters the network small.  The table is held at that scale through the solve, so
everything downstream is solved for what the lookup really delivers, and the report says
what the lookup was left with.  The measurements behind this are in :data:`DOCS`.

Backends mark every node that reads a tied table with the same ``meta["tied"]`` key.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .fan import EMBEDDING, fan_in
from .graph import ModelGraph, Node, NodeKind

DOCS = "https://jmiravet.github.io/AnyInit/experiments/tied-embeddings/"


@dataclass(frozen=True)
class TiedTable:
    """One table read both as a lookup and as an output layer."""

    lookups: tuple[str, ...]
    outputs: tuple[str, ...]
    width: int
    """The table's ``d_model``: the fan_in of its output role."""

    @property
    def scale(self) -> float:
        """The output role's scale, which the table takes."""
        return 1.0 / math.sqrt(self.width)


def find(graph: ModelGraph) -> tuple[TiedTable, ...]:
    """Tables marked as tied that the graph reads in both roles."""
    groups: dict[str, list[Node]] = {}
    for node in graph.of_kind(NodeKind.PARAMETRIC):
        key = node.meta.get("tied")
        if key is not None and node.spec is not None:
            groups.setdefault(str(key), []).append(node)

    tables = []
    for nodes in groups.values():
        lookups = tuple(n.id for n in nodes if n.spec is not None and n.spec.kind == EMBEDDING)
        outputs = [n for n in nodes if n.spec is not None and n.spec.is_fan_scaled]
        if lookups and outputs and outputs[0].spec is not None:
            width = int(fan_in(outputs[0].spec))
            tables.append(TiedTable(lookups, tuple(n.id for n in outputs), width))
    return tuple(tables)


def scales(tables: Sequence[TiedTable]) -> dict[str, float]:
    """The scale every node of every tied table is held at."""
    return {nid: table.scale for table in tables for nid in table.lookups + table.outputs}


def describe(table: TiedTable, graph: ModelGraph) -> str:
    """What was decided for one table, and why, for the report."""
    names = ", ".join(_name(graph, nid) for nid in table.lookups)
    outputs = ", ".join(_name(graph, nid) for nid in table.outputs)
    d = table.width
    return (
        f"tied embedding: {names} is also the output layer {outputs}. As an output it "
        f"needs 1/sqrt({d}) = {table.scale:.4g} for unit-variance logits, as a lookup it "
        f"would take 1; it was given the output's scale, so looked-up rows have "
        f"E[x^2] = {table.scale**2:.3g}. Unless forward() already multiplies them by "
        f"sqrt({d}), as Gemma and the original Transformer do, doing so restores unit "
        f"variance without touching the logits; see {DOCS}"
    )


def _name(graph: ModelGraph, node_id: str) -> str:
    return str(graph[node_id].meta.get("path") or node_id)
