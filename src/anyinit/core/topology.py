"""Pairing scalable layers with the activations they feed.

Follows graph edges rather than layer order, so branches that merge before an activation
are all recognized as feeding it.  In a ResNet transition block both the main and the
downsample normalization come back as ancestors of the one post-addition ReLU.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .graph import TRANSPARENT, ModelGraph, NodeKind


def measurement_node(graph: ModelGraph, scalable_id: str) -> str:
    """First activation downstream of a scalable node.

    Walks forward through pooling, dropout, reshapes and merges, stopping at the next
    scalable node.  Returns the node itself when nothing activates afterward.
    """
    best: int | None = None
    position = {nid: i for i, nid in enumerate(graph.order)}
    seen = {scalable_id}
    queue = deque(graph.successors(scalable_id))

    while queue:
        nid = queue.popleft()
        if nid in seen:
            continue
        seen.add(nid)
        node = graph[nid]
        if node.kind is NodeKind.ACTIVATION:
            idx = position[nid]
            if best is None or idx < best:
                best = idx
            continue
        if node.kind in TRANSPARENT:
            queue.extend(graph.successors(nid))
        # Scalable, input, output: stop. The signal belongs to another block now.

    return graph.order[best] if best is not None else scalable_id


def objective_node(graph: ModelGraph, activation_id: str) -> str:
    """Where an activation's target is enforced.

    At the end of the pooling, dropout and reshapes that follow it, since what governs
    propagation is the level the *next* layer receives.  Max pooling in particular changes
    that level substantially.  A branching path is ambiguous, so the search stops there.
    """
    current = activation_id
    while True:
        successors = graph.successors(current)
        if len(successors) != 1:
            return current
        nxt = graph[successors[0]]
        if nxt.kind not in TRANSPARENT or nxt.kind is NodeKind.MERGE:
            return current
        current = nxt.id


def dominating_scalable_ancestors(graph: ModelGraph, activation_id: str) -> tuple[str, ...]:
    """Scalable nodes whose scale controls what this activation receives.

    Walks backward through transparent nodes, collecting scalable nodes and stopping at any
    earlier activation.  Returns several when branches merge before the activation.
    """
    found: list[str] = []
    seen = {activation_id}
    queue = deque(graph.predecessors(activation_id))

    while queue:
        nid = queue.popleft()
        if nid in seen:
            continue
        seen.add(nid)
        node = graph[nid]
        if node.is_scalable:
            found.append(nid)
            continue
        if node.kind is NodeKind.ACTIVATION:
            continue
        if node.kind in TRANSPARENT:
            queue.extend(graph.predecessors(nid))

    position = {nid: i for i, nid in enumerate(graph.order)}
    return tuple(sorted(found, key=lambda n: position[n]))


def detect_shared(graph: ModelGraph) -> dict[str, tuple[str, ...]]:
    """Group nodes that share one underlying layer.

    A reused module appears once per call site but can carry only one scale.
    """
    buckets: dict[str, list[str]] = {}
    for node in graph.scalable:
        key = node.meta.get("shared_key")
        if key is None:
            continue
        buckets.setdefault(str(key), []).append(node.id)
    return {k: tuple(v) for k, v in buckets.items() if len(v) > 1}


def repair_path(
    graph: ModelGraph, activation_id: str, ancestors: tuple[str, ...]
) -> tuple[str, ...]:
    """Nodes whose moments change when an ancestor's scale changes, topologically ordered."""
    if not ancestors:
        return (activation_id,)

    downstream = set()
    queue = deque(ancestors)
    while queue:
        nid = queue.popleft()
        if nid in downstream:
            continue
        downstream.add(nid)
        if nid == activation_id:
            continue
        queue.extend(graph.successors(nid))

    upstream = set()
    queue = deque([activation_id])
    while queue:
        nid = queue.popleft()
        if nid in upstream:
            continue
        upstream.add(nid)
        if nid in ancestors:
            continue
        queue.extend(graph.predecessors(nid))

    inside = (downstream & upstream) | set(ancestors) | {activation_id}
    return tuple(nid for nid in graph.order if nid in inside)


@dataclass(frozen=True)
class Plan:
    """Precomputed pairings the solvers need.

    ``activation`` is which activation a layer feeds, and supplies the moment profile.
    ``measurement`` is where that activation's target is enforced, which sits further
    downstream when pooling or dropout follows; see :func:`objective_node`.
    """

    activation: dict[str, str]
    measurement: dict[str, str]
    objective: dict[str, str]
    ancestors: dict[str, tuple[str, ...]]
    repair: dict[str, tuple[str, ...]]
    shared: dict[str, tuple[str, ...]]

    def enforced_at(self, activation_id: str, *, substituted: bool = False) -> str:
        """Node where an activation's objective is checked.

        Normally past any pooling, where the next layer reads.  A substituted objective is
        defined on the activation's own range, so it is checked at the activation itself.
        """
        if substituted:
            return activation_id
        return self.objective.get(activation_id, activation_id)


def build_plan(graph: ModelGraph) -> Plan:
    """Resolve every layer/activation pairing in the graph once."""
    activation = {node.id: measurement_node(graph, node.id) for node in graph.scalable}
    ancestors = {
        node.id: dominating_scalable_ancestors(graph, node.id)
        for node in graph.of_kind(NodeKind.ACTIVATION)
    }
    objective = {aid: objective_node(graph, aid) for aid in ancestors}
    measurement = {scalable: objective.get(act, act) for scalable, act in activation.items()}
    repair = {
        aid: repair_path(graph, objective.get(aid, aid), anc) for aid, anc in ancestors.items()
    }
    return Plan(
        activation=activation,
        measurement=measurement,
        objective=objective,
        ancestors=ancestors,
        repair=repair,
        shared=detect_shared(graph),
    )
