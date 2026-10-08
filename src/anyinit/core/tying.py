"""Embedding tables that double as the output layer.

With tied weights one table is read twice.  As a lookup nothing sums over it, so its rows
enter the network at the table's own scale and want a scale of 1.  As the output layer it
sums over ``d_model`` inputs, so it wants ``1/sqrt(d_model)`` for unit-variance logits.
No single scale serves both.

The output role decides.  At the lookup's scale the logits' standard deviation is
``sqrt(d_model)`` and the initial loss grows with width; at the output's scale the lookup
merely enters the network small.  Constant factors the model applies around the output
layer count, so a model that scales its logits by ``1/sqrt(d_model)`` gets a table of
scale 1, which serves both roles.  The table is held at its scale through the solve, so
everything downstream is solved for what the lookup really delivers, and the report says
what the lookup was left with.  The measurements behind this are in :data:`DOCS`.

Backends mark every node that reads a tied table with the same ``meta["tied"]`` key.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
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
    output_factor: float = 1.0
    """Constant factor the model applies around the output layer, on either side of it."""
    lookup_factor: float = 1.0
    """Constant factor the model applies to the looked-up rows."""

    @property
    def scale(self) -> float:
        """The output role's scale, which the table takes."""
        return 1.0 / (self.output_factor * math.sqrt(self.width))

    @property
    def lookup_m2(self) -> float:
        """``E[x^2]`` of the looked-up rows, after the model's factor."""
        return (self.lookup_factor * self.scale) ** 2


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
            readout = outputs[0].id
            output_factor = _factor(graph, readout, graph.successors) * _factor(
                graph, readout, graph.predecessors
            )
            tables.append(
                TiedTable(
                    lookups,
                    tuple(n.id for n in outputs),
                    int(fan_in(outputs[0].spec)),
                    output_factor if output_factor > 0.0 else 1.0,
                    _factor(graph, lookups[0], graph.successors),
                )
            )
    return tuple(tables)


def scales(tables: Sequence[TiedTable]) -> dict[str, float]:
    """The scale every node of every tied table is held at."""
    return {nid: table.scale for table in tables for nid in table.lookups + table.outputs}


def describe(table: TiedTable, graph: ModelGraph) -> str:
    """One line for the report: what the table got and what the lookup delivers."""
    names = ", ".join(_name(graph, nid) for nid in table.lookups + table.outputs)
    return (
        f"tied embedding ({names}): scaled for the output layer ({table.scale:.4g}), so "
        f"the lookup delivers E[x^2] = {table.lookup_m2:.3g}; see {DOCS}"
    )


def _name(graph: ModelGraph, node_id: str) -> str:
    return str(graph[node_id].meta.get("path") or node_id)


def _factor(graph: ModelGraph, start: str, step: Callable[[str], tuple[str, ...]]) -> float:
    """Product of the constant factors met walking from ``start`` along a single path.

    Reshapes are passed through; anything else, or a branch, ends the walk.  A factor that
    comes with an offset is not a pure scaling and ends it too.
    """
    factor = 1.0
    current = start
    while len(step(current)) == 1:
        node = graph[step(current)[0]]
        if node.kind is NodeKind.SCALE and not node.meta.get("offset"):
            factor *= math.sqrt(node.meta["factor"].m2)
        elif node.kind is not NodeKind.SHAPE:
            break
        current = node.id
    return factor
