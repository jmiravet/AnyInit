"""The data-driven solver.

Pushes real batches through the model, measures what arrives, and corrects, with no
distributional assumption.

Layers are solved in topological order, which makes each exact: everything upstream is
already fixed, so the statistic at a layer's measurement point depends only on the scale
being solved.  That puts the cost at one forward pass per layer, and at exactly one for a
homogeneous activation, whose correction is a closed form.

The solver never touches a tensor.  It asks for statistics through ``measure`` and sets
scales through ``apply_scale``, both supplied by the backend.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence

from .graph import ModelGraph, NodeKind
from .moments import MomentState
from .profile import ActivationProfile
from .solve import Objective, SolveResult, resolve_objectives, sensitivity
from .topology import Plan

#: Measure statistics at the given node ids using the model's current weights.
MeasureFn = Callable[[Sequence[str]], dict[str, MomentState]]
#: Write a scale into the live model.
ApplyFn = Callable[[str, float], None]

MAX_ITERATIONS = 8
RELATIVE_TOL = 1e-3


def solve(
    graph: ModelGraph,
    plan: Plan,
    profiles: Mapping[str, ActivationProfile | None],
    *,
    measure: MeasureFn,
    apply_scale: ApplyFn,
    initial_scales: Mapping[str, float],
    fixed: Collection[str] = (),
    max_iterations: int = MAX_ITERATIONS,
) -> SolveResult:
    """Solve every scalable node's scale from measurements.

    Nodes in ``fixed`` keep their initial scale and are never adjusted.
    """
    result = SolveResult()
    result.scales = dict(initial_scales)
    result.objectives = resolve_objectives(graph, plan, profiles, fixed, result.warn)
    forwards = 0
    targeted = {a for ancestors in plan.ancestors.values() for a in ancestors}

    for node in graph.scalable:
        if node.id in fixed:
            continue
        activation = plan.activation[node.id]
        objective = result.objectives.get(activation)
        if objective is None or node.id not in targeted:
            # Nothing downstream sets a target: a layer feeding a normalization, or an
            # output layer.  The analytic mode leaves these at 1/sqrt(fan), and so does this.
            continue
        profile = profiles.get(activation)
        point = plan.enforced_at(activation, substituted=objective.substituted)
        forwards += _solve_layer(
            node.id, point, objective, profile, result, measure, apply_scale, max_iterations
        )

    activation_ids = [n.id for n in graph.of_kind(NodeKind.ACTIVATION)]
    scalable_ids = [n.id for n in graph.scalable]
    points = set(activation_ids) | set(scalable_ids) | set(plan.measurement.values())
    result.states = measure(sorted(points))
    result.iterations = forwards + 1
    result.converged = True
    return result


#: Largest factor a single update may move a scale by, so one bad measurement cannot fling
#: it out of range.
MAX_STEP = 10.0
#: Largest factor a scale may end up from where it started.  An objective that needs more
#: is unreachable for practical purposes, and is reported as such.
MAX_DRIFT = 1e3
#: Below this response of ``log E[signal]`` to ``log scale``, a layer no longer controls its
#: objective: whatever it does, other paths dominate the signal there.
FLAT_SLOPE = 0.02


def _solve_layer(
    node_id: str,
    point: str,
    objective: Objective,
    profile: ActivationProfile | None,
    result: SolveResult,
    measure: MeasureFn,
    apply_scale: ApplyFn,
    max_iterations: int,
) -> int:
    """Drive one layer's scale onto its objective; return the forward passes used.

    Steps in log space.  The first uses the activation's own sensitivity -- exact for a
    homogeneous activation -- and later ones the secant through the last two measurements,
    which absorbs anything the nominal sensitivity misses, such as the dilution a residual
    addition causes.  Every step and the total excursion are bounded, and the best scale
    seen is kept, so an unreachable objective ends in a warning rather than a runaway.
    """
    start = result.scales.get(node_id, 1.0)
    if start <= 0.0:
        return 0
    exponent = sensitivity(profile)
    log_scale = math.log(start)
    history: list[tuple[float, float, float]] = []
    best = (math.inf, log_scale)
    forwards = 0

    for _ in range(max_iterations):
        state = measure([point]).get(point)
        forwards += 1
        if state is None:
            result.warn(f"no measurement at {point!r} while solving {node_id!r}; scale left as is")
            break
        if not state.is_finite:
            result.warn(f"measurement at {point!r} went non-finite while solving {node_id!r}")
            break
        achieved = objective.of(state)
        if achieved <= 0.0:
            result.warn(f"signal vanished at {point!r} while solving {node_id!r}")
            break

        error = math.log(objective.value / achieved)
        if abs(error) < best[0]:
            best = (abs(error), log_scale)
        if abs(error) <= RELATIVE_TOL:
            break

        step = error / (2.0 * exponent)
        if history:
            previous_scale, previous_achieved, previous_error = history[-1]
            run = log_scale - previous_scale
            if abs(run) > 1e-12:
                slope = (math.log(achieved) - previous_achieved) / run
                if slope > FLAT_SLOPE:
                    step = error / slope
                elif error * previous_error > 0.0:
                    result.mark_unreachable(point)
                    result.warn(
                        f"objective at {point!r} is out of reach: scaling {node_id!r} no "
                        f"longer moves it from {achieved:.4g} toward {objective.value:.4g}, "
                        "typically because an identity skip already delivers that much"
                    )
                    break
        history.append((log_scale, math.log(achieved), error))

        step = max(-math.log(MAX_STEP), min(math.log(MAX_STEP), step))
        bounded = max(
            math.log(start) - math.log(MAX_DRIFT),
            min(math.log(start) + math.log(MAX_DRIFT), log_scale + step),
        )
        if bounded == log_scale:
            result.mark_unreachable(point)
            result.warn(
                f"objective at {point!r} is out of reach: {node_id!r} hit the "
                f"{MAX_DRIFT:g}x bound on its scale"
            )
            break
        log_scale = bounded
        result.scales[node_id] = math.exp(log_scale)
        apply_scale(node_id, result.scales[node_id])
    else:
        result.warn(f"scale for {node_id!r} did not settle within {max_iterations} iterations")

    if best[1] != log_scale:
        result.scales[node_id] = math.exp(best[1])
        apply_scale(node_id, result.scales[node_id])
    return forwards
