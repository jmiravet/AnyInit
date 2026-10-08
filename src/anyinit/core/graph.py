"""The intermediate representation every backend produces.

The contract between the framework-specific half of AnyInit and the mathematics.  A
backend walks whatever structure its framework exposes and emits nodes of these kinds;
everything downstream works on the IR alone.
"""

from __future__ import annotations

import enum
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .fan import ParamSpec


class NodeKind(enum.Enum):
    """What a node does to the signal passing through it."""

    INPUT = "input"
    OUTPUT = "output"
    PARAMETRIC = "parametric"
    NORMALIZATION = "normalization"
    ACTIVATION = "activation"
    POOL = "pool"
    MERGE = "merge"
    SCALE = "scale"
    """Multiplication by a constant the backend could read: ``meta["factor"]`` holds the
    constant's moments and ``meta["offset"]``, when present, a constant added after it."""
    DROPOUT = "dropout"
    SHAPE = "shape"
    OTHER = "other"


#: Nodes that carry a scale AnyInit can set.  Normalization belongs here: it resets the
#: signal's moments, so its gain controls what the next activation receives.
SCALABLE = frozenset({NodeKind.PARAMETRIC, NodeKind.NORMALIZATION})

#: Nodes a search walks straight through when pairing layers with activations.
TRANSPARENT = frozenset(
    {
        NodeKind.POOL,
        NodeKind.DROPOUT,
        NodeKind.SHAPE,
        NodeKind.MERGE,
        NodeKind.SCALE,
        NodeKind.OTHER,
    }
)


@dataclass(frozen=True)
class Node:
    """One operation in a traced model."""

    id: str
    kind: NodeKind
    op: str
    inputs: tuple[str, ...] = ()
    spec: ParamSpec | None = None
    handle: Any = None
    """Backend-private reference to the live layer or parameter.  Opaque to the core."""
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_scalable(self) -> bool:
        """Whether AnyInit can set a scale on this node."""
        return self.kind in SCALABLE

    def with_meta(self, **updates: Any) -> Node:
        """Return a copy with extra metadata merged in."""
        merged = dict(self.meta)
        merged.update(updates)
        return Node(self.id, self.kind, self.op, self.inputs, self.spec, self.handle, merged)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Node({self.id!r}, {self.kind.value}, {self.op!r})"


#: How completely a backend managed to observe the model.
FIDELITY_GRAPH = "graph"
FIDELITY_LINEAR = "linear"


class ModelGraph:
    """A traced model as a directed acyclic graph of :class:`Node`.

    ``fidelity`` is ``"graph"`` when branches and merges were observed and ``"linear"``
    when the backend could only recover an ordered chain, leaving residual connections
    invisible.  Reported rather than assumed.
    """

    def __init__(
        self,
        nodes: Sequence[Node],
        *,
        fidelity: str = FIDELITY_GRAPH,
        notes: Sequence[str] = (),
    ) -> None:
        self._nodes: dict[str, Node] = {}
        for node in nodes:
            if node.id in self._nodes:
                raise ValueError(f"duplicate node id {node.id!r}")
            self._nodes[node.id] = node
        self.order: tuple[str, ...] = tuple(n.id for n in nodes)
        self.fidelity = fidelity
        self.notes: tuple[str, ...] = tuple(notes)

        position = {nid: i for i, nid in enumerate(self.order)}
        self._successors: dict[str, list[str]] = {nid: [] for nid in self.order}
        for node in nodes:
            for src in node.inputs:
                if src not in self._nodes:
                    raise ValueError(f"node {node.id!r} refers to unknown input {src!r}")
                if position[src] >= position[node.id]:
                    raise ValueError(
                        f"nodes are not in topological order: {node.id!r} "
                        f"precedes its input {src!r}"
                    )
                self._successors[src].append(node.id)

    # -------------------------------------------------------------- accessors

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def __iter__(self) -> Iterator[Node]:
        return (self._nodes[nid] for nid in self.order)

    def __getitem__(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def get(self, node_id: str) -> Node | None:
        """Node by id, or ``None``."""
        return self._nodes.get(node_id)

    def successors(self, node_id: str) -> tuple[str, ...]:
        """Ids of the nodes this one feeds."""
        return tuple(self._successors[node_id])

    def predecessors(self, node_id: str) -> tuple[str, ...]:
        """Ids of the nodes feeding this one."""
        return self._nodes[node_id].inputs

    def of_kind(self, *kinds: NodeKind) -> tuple[Node, ...]:
        """Nodes of the given kinds, in topological order."""
        wanted = set(kinds)
        return tuple(self._nodes[nid] for nid in self.order if self._nodes[nid].kind in wanted)

    @property
    def scalable(self) -> tuple[Node, ...]:
        """Nodes carrying a scale AnyInit can set, in topological order."""
        return self.of_kind(*SCALABLE)

    def replace(self, node: Node) -> None:
        """Swap a node for an updated copy.  Ids, kinds and edges must match."""
        existing = self._nodes[node.id]
        if existing.inputs != node.inputs:
            raise ValueError(f"cannot change the edges of {node.id!r} after construction")
        self._nodes[node.id] = node
