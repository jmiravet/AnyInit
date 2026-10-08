"""Orchestration: turn a model plus a config into initialized weights and a report.

Trace, plan, profile, draw unit-variance weights, solve for one scale per layer, apply,
report.  The sequence is the same for every backend and every mode; only the trace and the
solve differ, and both sit behind an interface.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from .backends import Backend, resolve
from .config import InitConfig
from .core import analytic, distributions, empirical, tying
from .core.fan import ParamSpec, fan_in
from .core.graph import ModelGraph, Node, NodeKind
from .core.moments import MomentState
from .core.profile import ActivationProfile
from .core.registry import REGISTRY, ActivationRef
from .core.solve import SolveResult
from .core.stability import classify
from .core.topology import Plan, build_plan
from .report import InitReport, LayerRecord, StabilityRecord


def run(model: Any, config: InitConfig, params: Any = None) -> InitReport:
    """Initialize ``model`` according to ``config`` and describe what happened."""
    backend = resolve(model)
    backend.begin(model, params)

    graph = backend.build_graph(model, config.input_spec)
    plan = build_plan(graph)
    profiles = _profiles_for(graph, backend)
    fixed = _fixed_scales(graph, plan, config.gains)
    tied = tying.find(graph)
    # Scales the solve propagates but never changes.
    held = {**fixed, **tying.scales(tied)}

    weights = _Weights(backend, graph, config)
    weights.draw()

    input_state, measured_input = _input_state(backend, config)

    if config.mode == "analytic":
        result = analytic.solve(
            graph,
            plan,
            profiles,
            fixed=held,
            input_state=input_state,
            centered=config.center,
            distribution=config.distribution,
        )
        weights.apply_all(result.scales)
        measured, counts, problem = _validate(backend, model, graph, plan, config)
        if problem:
            result.warn(problem)
    else:
        problem = None
        initial = {n.id: analytic.default_scale(graph, n.id) for n in graph.scalable}
        initial.update(held)
        weights.apply_all(initial)
        inputs = _inputs(backend, config)
        result = empirical.solve(
            graph,
            plan,
            profiles,
            measure=lambda taps: _measure(backend, model, inputs, taps),
            apply_scale=weights.apply,
            initial_scales=initial,
            fixed=held.keys(),
        )
        weights.apply_all(result.scales)
        measured, counts = result.states, {}

    unscaled = backend.unscaled_weights(model, graph)
    report_params = backend.finalize(model)
    return _build_report(
        backend=backend,
        graph=graph,
        plan=plan,
        profiles=profiles,
        config=config,
        result=result,
        measured=measured,
        counts=counts,
        fixed=fixed,
        tied=tied,
        params=report_params,
        input_state=input_state,
        measured_input=measured_input,
        unscaled=unscaled,
        validation_error=problem,
    )


def _fixed_scales(graph: ModelGraph, plan: Plan, gains: Mapping[str, float]) -> dict[str, float]:
    """Scales of the layers feeding an activation whose gain the user fixed.

    Matched on the activation's name, so ``{"leaky_relu": g}`` covers every slope.
    """
    if not gains:
        return {}
    fixed: dict[str, float] = {}
    for node in graph.scalable:
        target = graph[plan.activation[node.id]]
        if target.kind is not NodeKind.ACTIVATION:
            continue
        name = _activation_ref(target).name
        if name in gains:
            fixed[node.id] = analytic.default_scale(graph, node.id, gains[name])
    return fixed


# --------------------------------------------------------------------- weights


class _Weights:
    """Unit-variance draws, kept so that applying a scale is a single multiply.

    Each new scale rewrites the same sample rather than resampling, so the solvers can
    iterate and ``seed`` fully determines the result.
    """

    def __init__(self, backend: Backend, graph: ModelGraph, config: InitConfig) -> None:
        self._backend = backend
        self._graph = graph
        self._config = config
        self._samples: dict[str, np.ndarray] = {}

    def draw(self) -> None:
        cfg = self._config
        seed = 0 if cfg.seed is None else cfg.seed
        for index, node in enumerate(self._graph.scalable):
            if node.handle is None or node.spec is None:
                continue
            if node.kind is NodeKind.NORMALIZATION:
                continue
            # One stream per layer, so adding a layer does not reshuffle the others.
            rng = np.random.default_rng([seed, index])
            self._samples[node.id] = distributions.sample(
                node.spec, cfg.distribution, rng, center=cfg.center
            )
            if node.spec.has_bias:
                self._backend.write_bias(node.handle, np.zeros(node.spec.out_units))

    def apply(self, node_id: str, scale: float) -> None:
        node = self._graph[node_id]
        if node.handle is None:
            return
        if node.kind is NodeKind.NORMALIZATION:
            self._backend.write_gain(node.handle, scale)
        else:
            sample = self._samples.get(node_id)
            if sample is None:
                return
            self._backend.write_weight(node.handle, sample * scale)

    def apply_all(self, scales: Mapping[str, float]) -> None:
        for node in self._graph.scalable:
            scale = scales.get(node.id)
            if scale is not None:
                self.apply(node.id, scale)


# ------------------------------------------------------------------ measuring


def _input_state(backend: Backend, config: InitConfig) -> tuple[MomentState, bool]:
    """Boundary condition for the recursion, and whether it came from real data.

    A shape means AnyInit synthesizes the batch, so a standard normal is assumed.  A
    supplied batch's moments are read off it instead, since assuming otherwise would
    rescale the whole network.  The analytic solve still never runs the model.
    """
    spec = config.input_spec
    if spec is None or (isinstance(spec, (tuple, list)) and all(isinstance(d, int) for d in spec)):
        return MomentState.standard_normal(), False
    try:
        batch = spec() if callable(spec) else spec
        state = backend.input_moments(batch)
    except Exception:
        state = None
    if state is None or not state.is_finite or state.m2 <= 0.0:
        return MomentState.standard_normal(), False
    return state, True


def _inputs(backend: Backend, config: InitConfig) -> Any:
    spec = config.input_spec
    if callable(spec) and not isinstance(spec, (tuple, list)):
        return spec
    return backend.make_inputs(spec, config.seed)


def _measure(
    backend: Backend, model: Any, inputs: Any, taps: Sequence[str]
) -> dict[str, MomentState]:
    """Moments at ``taps`` over one batch, a fresh one each time ``inputs`` is callable."""
    return backend.forward_taps(model, inputs() if callable(inputs) else inputs, taps)


#: Most rows the validation pass uses when it synthesizes its own batch.  A second-moment
#: estimate over a few hundred rows is noisy enough to dominate the comparison.
_VALIDATION_ROWS = 4096
#: Cap on the elements of a synthesized validation batch, so that widening it for
#: statistics never makes the forward pass the memory problem.  2**21 float32 values is
#: 8 MB of input: thousands of rows of a small input, a dozen 224x224 images.
_VALIDATION_ELEMENTS = 1 << 21


def _validation_spec(input_spec: Any) -> Any:
    """Size a synthetic validation batch within a fixed element budget.

    Applies only when ``input_spec`` is a shape; supplied data is used as given.  Small
    inputs get up to ``_VALIDATION_ROWS`` rows for tight statistics; large ones get as
    many rows as ``_VALIDATION_ELEMENTS`` allows, and never fewer than one.
    """
    if (
        isinstance(input_spec, (tuple, list))
        and len(input_spec) >= 2
        and all(isinstance(d, int) for d in input_spec)
    ):
        per_row = max(math.prod(int(d) for d in input_spec[1:]), 1)
        rows = max(1, min(_VALIDATION_ROWS, _VALIDATION_ELEMENTS // per_row))
        return (rows, *input_spec[1:])
    return input_spec


def _validate(
    backend: Backend, model: Any, graph: ModelGraph, plan: Plan, config: InitConfig
) -> tuple[dict[str, MomentState], dict[str, int], str | None]:
    """One measured forward pass, purely to check the analytic prediction.

    Never used to choose a scale; it only populates the report's validation line.  A
    failure does not abort the initialization -- the weights are already set -- but it is
    returned as a message for the report rather than dropped.
    """
    if config.input_spec is None:
        return {}, {}, None
    taps = (
        [n.id for n in graph.of_kind(NodeKind.ACTIVATION)]
        + [n.id for n in graph.scalable]
        + list(plan.measurement.values())
    )
    try:
        widened = replace(config, input_spec=_validation_spec(config.input_spec))
        measured = _measure(backend, model, _inputs(backend, widened), sorted(set(taps)))
    except Exception as exc:
        return {}, {}, f"validation skipped: {type(exc).__name__}: {exc}".splitlines()[0]
    return measured, dict(getattr(backend, "last_counts", {}) or {}), None


# ------------------------------------------------------------------- profiles


def _profiles_for(graph: ModelGraph, backend: Backend) -> dict[str, ActivationProfile | None]:
    out: dict[str, ActivationProfile | None] = {}
    for node in graph.of_kind(NodeKind.ACTIVATION):
        out[node.id] = REGISTRY.profile(_activation_ref(node), backend)
    return out


# --------------------------------------------------------------------- report


def _activation_ref(node: Node) -> ActivationRef:
    return node.meta.get("activation") or ActivationRef(node.op)


def _name_of(node: Node) -> str:
    return str(node.meta.get("path") or node.id)


def _build_report(
    *,
    backend: Backend,
    graph: ModelGraph,
    plan: Plan,
    profiles: dict[str, ActivationProfile | None],
    config: InitConfig,
    result: SolveResult,
    measured: dict[str, MomentState],
    counts: dict[str, int],
    fixed: Mapping[str, float],
    tied: Sequence[tying.TiedTable],
    params: Any,
    input_state: MomentState,
    measured_input: bool,
    unscaled: list[tuple[str, str]],
    validation_error: str | None,
) -> InitReport:
    upstream = _upstream_draws(graph, plan, result)
    layers: list[LayerRecord] = []
    for node in graph.scalable:
        activation = plan.activation.get(node.id, node.id)
        point = _enforced_at(plan, result, node.id)
        profile = profiles.get(activation)
        spec: ParamSpec | None = node.spec
        fan = fan_in(spec) if spec is not None and spec.is_fan_scaled else None
        predicted = result.states.get(point)
        notes = (str(node.meta["note"]),) if "note" in node.meta else ()
        layers.append(
            LayerRecord(
                name=_name_of(node),
                kind=node.op,
                activation=_activation_label(graph, point, node.id, profile),
                fan=fan,
                scale=result.scales.get(node.id),
                is_gain=node.kind is NodeKind.NORMALIZATION,
                predicted=None if predicted is None else predicted.m2,
                measured=None if point not in measured else measured[point].m2,
                measured_count=counts.get(point),
                units=None if spec is None else spec.out_units,
                constrained=point in result.objectives,
                fixed=node.id in fixed,
                notes=notes,
                upstream_draws=upstream.get(node.id, 0.0),
            )
        )

    depth = _activation_depth(graph)
    stability: dict[str, StabilityRecord] = {}
    for profile in profiles.values():
        if profile is None or profile.name in stability:
            continue
        diag = profile.diagnostics
        stability[profile.name] = StabilityRecord(
            activation=profile.name,
            chi=diag.chi,
            sigma_star=diag.sigma_star,
            homogeneous_degree=diag.homogeneous_degree,
            feasible_m2=diag.feasible_m2,
            verdict=classify(diag.chi, diag.sigma_star, depth),
            depth=depth,
        )

    warnings = list(result.warnings)
    if unscaled:
        warnings.append(
            f"{len(unscaled)} weight tensor(s) left as the framework initialized them; "
            "see the 'Not scaled' section"
        )
    if measured_input:
        warnings.append(
            "the analytic recursion was started from the supplied batch's own moments "
            f"(E[x]={input_state.mean:.4g}, E[x^2]={input_state.m2:.4g}) rather than "
            "assuming a standard normal input"
        )
    present = {_activation_ref(n).name for n in graph.of_kind(NodeKind.ACTIVATION)}
    warnings.extend(
        f"gains: no activation named {name!r} in this model, so that gain was not used"
        for name in config.gains
        if name not in present
    )
    warnings.extend(graph.notes)
    warnings.extend(config.notes)
    warnings.extend(
        f"one layer is used at {len(ids)} call sites ({', '.join(ids)}); it can only "
        "carry one scale, so the last one solved wins"
        for ids in plan.shared.values()
    )
    warnings.extend(tying.describe(table, graph) for table in tied)
    missing = [nid for nid, profile in profiles.items() if profile is None]
    if missing:
        warnings.append(
            f"{len(missing)} activation node(s) have no moment profile and were treated as "
            f"shape-preserving: {', '.join(sorted(missing)[:6])}"
        )

    return InitReport(
        backend=backend.label,
        mode=config.mode,
        graph_fidelity=graph.fidelity,
        config=config.summary(),
        layers=tuple(layers),
        stability=tuple(stability.values()),
        warnings=tuple(dict.fromkeys(warnings)),
        iterations=result.iterations,
        converged=result.converged,
        objective_error=result.objective_error,
        params=params,
        unscaled=tuple(unscaled),
        validation_error=validation_error,
    )


def _upstream_draws(graph: ModelGraph, plan: Plan, result: SolveResult) -> dict[str, float]:
    """Relative variance that upstream weight draws add to each scalable layer's level.

    For one input, a layer's draw scales the second moment of its activation's output by
    a factor whose relative variance is ``(E[a^4]/E[a^2]^2 - 1) / units``: 2/units for a
    linear layer, 5/units for ReLU.  Those factors multiply along a path, so their
    variances add, and a normalization starts the count again.  A merge takes its deepest
    branch.
    """
    carried: dict[str, float] = {}
    upstream: dict[str, float] = {}
    for nid in graph.order:
        node = graph[nid]
        inherited = max((carried[p] for p in graph.predecessors(nid)), default=0.0)
        if node.kind is NodeKind.NORMALIZATION:
            inherited = 0.0
        elif node.is_scalable and node.spec is not None and node.spec.out_units:
            upstream[nid] = inherited
            state = result.states.get(_enforced_at(plan, result, nid))
            kurtosis = state.kurtosis_of_square if state is not None and state.m2 > 0 else 3.0
            inherited += max(kurtosis - 1.0, 0.0) / node.spec.out_units
        carried[nid] = inherited
    return upstream


def _activation_label(
    graph: ModelGraph, point: str, scalable_id: str, profile: ActivationProfile | None
) -> str:
    """Describe what this layer feeds, with the intervening nodes shown.

    An activation's name alone would hide that two normalizations reach the same
    post-addition activation; prefixing the hops makes ``add->relu`` legible on both rows.
    """
    chain = _chain_label(graph, point, scalable_id)
    if profile is None:
        return chain
    if chain in ("none", profile.name):
        return profile.name
    return chain


def _enforced_at(plan: Plan, result: SolveResult, scalable_id: str) -> str:
    """Node where the objective governing a layer is checked."""
    activation = plan.activation.get(scalable_id, scalable_id)
    objective = result.objectives.get(activation)
    return plan.enforced_at(activation, substituted=objective is not None and objective.substituted)


def _chain_label(graph: ModelGraph, point: str, scalable_id: str) -> str:
    """Describe what sits between a layer and the point its target is enforced at.

    Walks backwards from the enforcement point and stops at a merge, past which more than
    one branch would have to be described.
    """
    if point == scalable_id:
        return "none"

    hops: list[str] = []
    current = point
    while current != scalable_id and len(hops) < 4:
        node = graph[current]
        if node.kind is not NodeKind.SHAPE:
            hops.append(node.op)
        if node.kind is NodeKind.MERGE:
            break
        preds = graph.predecessors(current)
        if not preds:
            break
        current = preds[0]
    return "\u2192".join(reversed(hops)) if hops else "none"


def _activation_depth(graph: ModelGraph) -> int:
    return max(len(graph.of_kind(NodeKind.ACTIVATION)), 1)
