"""Shared vocabulary for the two solvers."""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field

from .graph import ModelGraph, NodeKind
from .moments import MomentState
from .profile import ActivationProfile
from .topology import Plan

#: Clamp on the sensitivity exponent, so a pathological local slope cannot turn one
#: update into a wild jump.
MIN_EXPONENT = 0.25
MAX_EXPONENT = 8.0
#: A target this close to an activation's ceiling counts as unreachable.
_SATURATION_MARGIN = 0.98


#: Which moment an objective constrains.
METRIC_M2 = "m2"
METRIC_VAR = "var"


@dataclass(frozen=True)
class Objective:
    """The condition one activation's upstream layers are solved against.

    Most activations get a second-moment target.  Saturating ones cannot reach it --
    ``tanh`` approaches ``E[a^2] = 1`` only asymptotically, ``sigmoid`` tops out at 0.5 --
    so for those the metric switches to variance and the target becomes the middle of the
    reachable range.  Well-posed for every bounded activation, and always reported.
    """

    metric: str = METRIC_M2
    value: float = 1.0
    substituted: bool = False

    def of(self, state: MomentState) -> float:
        """Read whichever moment this objective constrains."""
        return state.var if self.metric == METRIC_VAR else state.m2


@dataclass
class SolveResult:
    """What a solver decided, and what it could not do."""

    scales: dict[str, float] = field(default_factory=dict)
    states: dict[str, MomentState] = field(default_factory=dict)
    objectives: dict[str, Objective] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    iterations: int = 0
    converged: bool = True
    objective_error: float = 0.0
    unreachable: tuple[str, ...] = ()
    """Checkpoints whose objective no choice of scale can reach, typically because an
    identity skip already delivers more than the target allows.  Not a convergence
    failure: the solve is finished, with nothing left to adjust."""

    def mark_unreachable(self, node_id: str) -> None:
        """Record a checkpoint whose objective cannot be met."""
        if node_id not in self.unreachable:
            self.unreachable = (*self.unreachable, node_id)

    def warn(self, message: str) -> None:
        """Record a message, ignoring duplicates."""
        if message not in self.warnings:
            self.warnings = (*self.warnings, message)


def sensitivity(
    profile: ActivationProfile | None,
    *,
    mu: float = 0.0,
    var: float | None = None,
) -> float:
    """How strongly a layer's scale moves its activation's second moment.

    ``E[f(z)^2]`` behaves like ``(sigma_w^2)^chi`` near the operating point, so correcting
    the second moment by ``r`` needs ``r^(1/2chi)`` on the scale.  Exact for a homogeneous
    activation with ``chi = p``; a Newton step in log space otherwise.

    ``var`` gives the slope where the solver currently stands rather than the global
    figure, which matters for a saturating activation whose slope collapses as it
    saturates.
    """
    if profile is None:
        return 1.0
    if var is not None and var > 0.0:
        chi = profile.local_slope(mu, var)
    else:
        degree = profile.homogeneous_degree
        chi = float(degree) if degree is not None and degree > 0.0 else profile.chi
    if not math.isfinite(chi) or chi <= 0.0:
        return 1.0
    return min(max(chi, MIN_EXPONENT), MAX_EXPONENT)


def update_scale(scale: float, ratio: float, exponent: float) -> float:
    """One multiplicative correction step, in log space."""
    if not math.isfinite(ratio) or ratio <= 0.0:
        return scale
    step = ratio ** (1.0 / (2.0 * exponent))
    updated = float(scale * step)
    if not math.isfinite(updated) or updated <= 0.0:
        return scale
    return updated


def resolve_objectives(
    graph: ModelGraph,
    plan: Plan,
    profiles: Mapping[str, ActivationProfile | None],
    fixed: Collection[str],
    report: Callable[[str], None],
) -> dict[str, Objective]:
    """Decide what each activation is driven toward: ``E[a^2] = 1``.

    Shared by both solvers, so that switching mode does not change the target.  An
    activation whose layers all carry a fixed gain has nothing left to solve, and so no
    objective.
    """
    objectives: dict[str, Objective] = {}
    for node in graph.of_kind(NodeKind.ACTIVATION):
        profile = profiles.get(node.id)
        if profile is None:
            continue
        ancestors = plan.ancestors.get(node.id, ())
        if ancestors and all(a in fixed for a in ancestors):
            continue
        ceiling = profile.feasible_m2[1]
        if math.isfinite(ceiling) and ceiling * _SATURATION_MARGIN < 1.0:
            _, high = profile.feasible_var
            substitute = 0.5 * high if math.isfinite(high) else 1.0
            objectives[node.id] = Objective(METRIC_VAR, substitute, substituted=True)
            report(
                f"activation {profile.name!r} saturates: E[a^2] cannot exceed "
                f"{ceiling:.4g}, so a target of 1 is unreachable. "
                f"Aiming at Var[a]={substitute:.4g}, the middle of its reachable range"
            )
        else:
            objectives[node.id] = Objective()
    return objectives
