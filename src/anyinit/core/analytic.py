"""The data-free solver.

Propagates moments through the graph and solves for one scale per layer without running
the model.

The traversal is Gauss-Seidel: each activation is corrected the moment the walk reaches
it, when everything feeding its layers is already settled and the only unknown left is the
scale being solved.  One pass then suffices for a feed-forward network, and for a
homogeneous activation the local correction is a closed form that lands in a single step.
A global sweep would instead propagate information one layer at a time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from . import transfer
from .distributions import is_centered
from .fan import EMBEDDING, fan_in
from .graph import ModelGraph, Node, NodeKind
from .moments import MomentState
from .profile import ActivationProfile
from .solve import (
    MIN_EXPONENT,
    Objective,
    SolveResult,
    resolve_objectives,
    sensitivity,
    update_scale,
)
from .topology import Plan

#: Local Newton steps allowed per activation.  Homogeneous activations need one.
MAX_LOCAL_STEPS = 12
RELATIVE_TOL = 1e-12
#: Passes over the whole graph.  A feed-forward network settles in one; merges couple
#: branches that the walk reaches at different times.
MAX_GLOBAL_PASSES = 12
#: A solve counts as converged when every objective sits this close to its target.
#: Judged on whether the objectives are met rather than on whether the scales stopped
#: moving, so a correctly scaled network is not reported as a failure.
#:
#: Convergence threshold, set to what is physically meaningful rather than to machine
#: precision: residual coupling leaves branches constraining each other, and half a
#: percent on a second moment sits below the noise of any measurement of it.  The exact
#: residual is reported regardless.
OBJECTIVE_TOL = 5e-3


def solve(
    graph: ModelGraph,
    plan: Plan,
    profiles: Mapping[str, ActivationProfile | None],
    *,
    fixed: Mapping[str, float] | None = None,
    input_state: MomentState | None = None,
    centered: bool = False,
    distribution: str = "normal",
) -> SolveResult:
    """Solve for the scale of every scalable node.

    Nodes in ``fixed`` keep the scale given there and are propagated, never solved.
    """
    fixed = dict(fixed or {})
    result = SolveResult()
    result.scales = {n.id: default_scale(graph, n.id) for n in graph.scalable}
    result.scales.update(fixed)

    context = _Context(
        graph=graph,
        plan=plan,
        profiles=profiles,
        fixed=set(fixed),
        centered=centered,
        distribution=distribution,
        input_state=input_state or MomentState.standard_normal(),
        result=result,
    )
    objectives = resolve_objectives(graph, plan, profiles, set(fixed), result.warn)
    # Objectives are declared per activation but checked where they are enforced.
    checkpoints = {
        plan.enforced_at(aid, substituted=obj.substituted): obj for aid, obj in objectives.items()
    }

    passes = 0
    error = math.inf
    for attempt in range(1, MAX_GLOBAL_PASSES + 1):
        passes = attempt
        states = context.walk(objectives)
        context.flag_unreachable(objectives, states)
        error = max_objective_error(states, checkpoints, skip=set(result.unreachable))
        if error < OBJECTIVE_TOL:
            break

    result.states = states
    result.objectives = checkpoints
    result.iterations = passes
    result.converged = error < OBJECTIVE_TOL
    result.objective_error = error
    if not result.converged:
        worst = _worst_objective(states, checkpoints)
        result.warn(
            f"analytic solve left {worst} off its objective by {error:.2e} after "
            f"{MAX_GLOBAL_PASSES} passes; using the best scales found"
        )
    return result


def max_objective_error(
    states: Mapping[str, MomentState],
    objectives: Mapping[str, Objective],
    *,
    skip: set[str] | None = None,
) -> float:
    """Largest relative gap between an objective and what is achieved.

    Checkpoints in ``skip`` have unreachable objectives and are excluded.
    """
    skip = skip or set()
    worst = 0.0
    for node_id, objective in objectives.items():
        if node_id in skip:
            continue
        state = states.get(node_id)
        if state is None or not state.is_finite or objective.value <= 0.0:
            return math.inf
        worst = max(worst, abs(objective.of(state) - objective.value) / objective.value)
    return worst


def _worst_objective(states: Mapping[str, MomentState], objectives: Mapping[str, Objective]) -> str:
    worst, name = -1.0, "?"
    for node_id, objective in objectives.items():
        state = states.get(node_id)
        if state is None or objective.value <= 0.0:
            continue
        gap = abs(objective.of(state) - objective.value) / objective.value
        if gap > worst:
            worst, name = gap, node_id
    return repr(name)


def default_scale(graph: ModelGraph, node_id: str, gain: float = 1.0) -> float:
    """Return ``gain/sqrt(fan_in)``, or ``gain`` itself where there is no fan to divide by.

    A normalization's output has unit variance and an embedding's is its table's, so for
    both the gain is the scale.
    """
    node = graph[node_id]
    spec = node.spec
    if node.kind is NodeKind.NORMALIZATION or spec is None or not spec.is_fan_scaled:
        return gain
    return gain / math.sqrt(max(fan_in(spec), 1.0))


@dataclass
class _Context:
    """Carries the solve's settings and mutable result through the walk."""

    graph: ModelGraph
    plan: Plan
    profiles: Mapping[str, ActivationProfile | None]
    fixed: set[str]
    distribution: str
    centered: bool
    input_state: MomentState
    result: SolveResult

    # --------------------------------------------------------------- traversal

    def walk(self, objectives: Mapping[str, Objective]) -> dict[str, MomentState]:
        """One Gauss-Seidel pass: propagate, correcting at each checkpoint on arrival.

        The correction fires at the node where an objective is *enforced*, which sits
        downstream of the activation it belongs to whenever pooling follows.
        """
        graph, plan = self.graph, self.plan
        triggers = {
            plan.enforced_at(aid, substituted=obj.substituted): aid
            for aid, obj in objectives.items()
        }
        states: dict[str, MomentState] = {}

        for node in graph:
            states[node.id] = self.evaluate(node, states)
            activation_id = triggers.get(node.id)
            if activation_id is not None:
                self.correct(activation_id, objectives[activation_id], states)
        return states

    def correct(
        self, activation_id: str, objective: Objective, states: dict[str, MomentState]
    ) -> None:
        """Drive one activation onto its target by adjusting the layers that feed it."""
        plan = self.plan
        graph = self.graph
        result = self.result
        profile = self.profiles.get(activation_id)

        ancestors = [a for a in plan.ancestors.get(activation_id, ()) if a not in self.fixed]
        if profile is None:
            return
        if not ancestors:
            # Nothing upstream carries a scale: every path into this activation already
            # passes through another one.  Its output is whatever the merge produces, and
            # saying so beats silently leaving it out of the solve.
            result.warn(
                f"activation {activation_id!r} has no scalable layer feeding it (every "
                "path into it crosses another activation), so its output level is not "
                "something AnyInit can set"
            )
            return

        path = plan.repair[activation_id]
        node = graph[activation_id]
        point = plan.enforced_at(activation_id, substituted=objective.substituted)

        for _ in range(MAX_LOCAL_STEPS):
            state = states.get(point)
            pre = states.get(node.inputs[0]) if node.inputs else None
            if state is None or pre is None:
                result.warn(
                    f"could not evaluate {point!r} while solving {activation_id!r}; "
                    "leaving its layers as drawn"
                )
                return
            if not state.is_finite:
                result.warn(f"moments at {point!r} went non-finite; leaving its layers as drawn")
                return

            achieved = objective.of(state)
            if achieved <= 0.0:
                result.warn(f"signal vanishes at {activation_id!r}; leaving its layers as drawn")
                return
            ratio = objective.value / achieved
            if abs(ratio - 1.0) <= RELATIVE_TOL:
                return

            base = sensitivity(profile, mu=pre.mean, var=pre.var)
            # After a merge a branch owns only part of the signal, so its scale has less
            # leverage than the activation's own slope suggests.
            shares = self._merge_shares(activation_id, ancestors, states)
            moved = False
            for ancestor in ancestors:
                current = result.scales.get(ancestor, 1.0)
                exponent = max(base * shares.get(ancestor, 1.0), MIN_EXPONENT)
                updated = update_scale(current, ratio, exponent)
                if updated != current:
                    result.scales[ancestor] = updated
                    moved = True
            if not moved:
                self._report_stuck(activation_id, objective, achieved, ancestors, states)
                return

            for nid in path:
                states[nid] = self.evaluate(graph[nid], states)
        self._report_stuck(point, objective, objective.of(states[point]), ancestors, states)

    def flag_unreachable(
        self, objectives: Mapping[str, Objective], states: Mapping[str, MomentState]
    ) -> None:
        """Record objectives that no choice of scale can satisfy.

        An activation fed by a path with no adjustable scale on it receives that path's
        signal regardless.  When that alone exceeds the target the solve is complete
        rather than failed, and the report says so.
        """
        plan = self.plan
        result = self.result
        for activation_id, objective in objectives.items():
            point = plan.enforced_at(activation_id, substituted=objective.substituted)
            if point in result.unreachable:
                continue
            floor = self._uncontrollable_floor(activation_id, states)
            if floor is None:
                continue
            state = states.get(point)
            if state is None:
                continue
            # Translate the floor through the activation itself: what reaches the
            # checkpoint is f(skip), not the skip.
            profile = self.profiles.get(activation_id)
            delivered = profile.moments(0.0, floor).m2 if profile is not None else floor
            if delivered > objective.value * (1.0 + OBJECTIVE_TOL):
                result.mark_unreachable(point)
                result.warn(
                    f"activation {activation_id!r} already receives {delivered:.4g} from "
                    f"paths with no scalable layer on them, above its objective of "
                    f"{objective.value:.4g}. The upstream scales were driven to their floor"
                )

    def _merge_shares(
        self,
        activation_id: str,
        ancestors: Sequence[str],
        states: Mapping[str, MomentState],
    ) -> dict[str, float]:
        """Fraction of the activation's input each ancestor's branch supplies."""
        graph = self.graph
        node = graph[activation_id]
        if not node.inputs:
            return dict.fromkeys(ancestors, 1.0)
        merge = graph[node.inputs[0]]
        if merge.kind is not NodeKind.MERGE or len(merge.inputs) < 2:
            return dict.fromkeys(ancestors, 1.0)

        totals = {}
        for source in merge.inputs:
            state = states.get(source)
            totals[source] = state.m2 if state is not None else 0.0
        grand = sum(totals.values())
        if grand <= 0.0:
            return dict.fromkeys(ancestors, 1.0)

        shares = {}
        for ancestor in ancestors:
            owned = sum(
                value for source, value in totals.items() if _descends_from(graph, source, ancestor)
            )
            shares[ancestor] = max(owned / grand, 1e-3)
        return shares

    def _report_stuck(
        self,
        activation_id: str,
        objective: Objective,
        achieved: float,
        ancestors: Sequence[str],
        states: Mapping[str, MomentState],
    ) -> None:
        """Record why an activation could not be driven onto its objective."""
        result = self.result
        if abs(achieved / objective.value - 1.0) <= 0.05:
            return

        floor = self._uncontrollable_floor(activation_id, states)
        if floor is not None and floor > objective.value:
            result.warn(
                f"activation {activation_id!r} receives {floor:.3g} from paths with no "
                f"scalable layer on them (an identity skip, typically), which already "
                f"exceeds the objective of {objective.value:.3g}. Scaling "
                f"{', '.join(repr(a) for a in ancestors)} cannot bring it down"
            )
        else:
            result.warn(
                f"activation {activation_id!r} settled at {achieved:.4g} against an "
                f"objective of {objective.value:.4g}"
            )

    def _uncontrollable_floor(
        self, activation_id: str, states: Mapping[str, MomentState]
    ) -> float | None:
        """Signal reaching an activation along paths it has no scale on.

        A path is uncontrollable when none of this activation's dominating ancestors lies
        on it.  Scalable layers further upstream do not count: each is pinned by its own
        objective already.
        """
        graph, plan = self.graph, self.plan
        node = graph[activation_id]
        if not node.inputs:
            return None
        merge = graph[node.inputs[0]]
        if merge.kind is not NodeKind.MERGE:
            return None

        ancestors = plan.ancestors.get(activation_id, ())
        total = 0.0
        found = False
        for source in merge.inputs:
            if any(_descends_from(graph, source, a) for a in ancestors):
                continue
            state = states.get(source)
            if state is not None:
                total += state.m2
                found = True
        return total if found else None

    # -------------------------------------------------------------- evaluation

    def evaluate(self, node: Node, states: Mapping[str, MomentState]) -> MomentState:
        """Moments at one node, given its predecessors."""
        if node.kind is NodeKind.INPUT:
            return self.input_state

        incoming = [states[src] for src in node.inputs if src in states]
        # Some frameworks expose no input node, leaving the first real layer without a
        # predecessor.  It still applies its own transfer, so that its scale is visible to
        # the solver.
        state_in = incoming[0] if incoming else self.input_state
        scales = self.result.scales
        kind = node.kind

        if kind is NodeKind.PARAMETRIC:
            spec = node.spec
            if spec is None:
                return state_in
            sigma = scales.get(node.id, 1.0)
            if spec.kind == EMBEDDING:
                return transfer.through_embedding(sigma)
            return transfer.through_linear(
                state_in, spec, sigma, centered=is_centered(self.distribution, spec, self.centered)
            )

        if kind is NodeKind.NORMALIZATION:
            return transfer.through_normalization(scales.get(node.id, 1.0))

        if kind is NodeKind.ACTIVATION:
            profile = self.profiles.get(node.id)
            if profile is None:
                return state_in
            layer_fan, layer_sigma = self._upstream_layer(node.id, scales)

            return transfer.through_activation(
                state_in,
                profile,
                fan=layer_fan,
                sigma_w=layer_sigma,
                centered=self.centered,
            )

        if kind is NodeKind.MERGE:
            return transfer.through_merge(incoming, node.op)

        if kind is NodeKind.SCALE:
            offset = float(node.meta.get("offset", 0.0))
            return transfer.through_scale(state_in, node.meta["factor"], offset)

        if kind is NodeKind.DROPOUT:
            return transfer.through_dropout(state_in, float(node.meta.get("p", 0.0)))

        if kind is NodeKind.POOL:
            window = int(node.meta.get("window", 1))
            if node.meta.get("pool") == "max":
                exact = self._max_through_activation(node, window, states)
                if exact is not None:
                    return exact
                return transfer.through_max_pool(state_in, window)
            return transfer.through_average_pool(state_in, window)

        return state_in

    def _max_through_activation(
        self, node: Node, window: int, states: Mapping[str, MomentState]
    ) -> MomentState | None:
        """Max pooling directly after an activation, done exactly.

        ``None`` when the predecessor is not a profiled activation, leaving the caller the
        matched-Gaussian approximation.
        """
        graph = self.graph
        if window <= 1 or not node.inputs:
            return None
        previous = graph[node.inputs[0]]
        if previous.kind is not NodeKind.ACTIVATION:
            return None
        profile = self.profiles.get(previous.id)
        if profile is None or not previous.inputs:
            return None
        pre = states.get(previous.inputs[0])
        if pre is None:
            return None
        return profile.max_moments(pre.mean, pre.var, float(window))

    def _upstream_layer(
        self, activation_id: str, scales: Mapping[str, float]
    ) -> tuple[float | None, float | None]:
        """Fan and scale of the one layer feeding an activation, when there is one.

        Needed only by the scale-mixture correction, which does not apply once several
        branches contribute.
        """
        ancestors = self.plan.ancestors.get(activation_id, ())
        if len(ancestors) != 1:
            return None, None
        node = self.graph[ancestors[0]]
        if node.kind is not NodeKind.PARAMETRIC or node.spec is None or not node.spec.is_fan_scaled:
            return None, None
        return fan_in(node.spec), scales.get(node.id, 1.0)


# ----------------------------------------------------------------- helpers


def _descends_from(graph: ModelGraph, node_id: str, ancestor_id: str) -> bool:
    """Whether ``ancestor_id`` lies upstream of ``node_id``."""
    from collections import deque

    seen = {node_id}
    queue = deque([node_id])
    while queue:
        current = queue.popleft()
        if current == ancestor_id:
            return True
        for source in graph[current].inputs:
            if source not in seen:
                seen.add(source)
                queue.append(source)
    return False
