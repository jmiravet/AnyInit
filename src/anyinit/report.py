"""What AnyInit did, and what it could not do.

An initialization can be wrong in ways that only surface as a network that will not train.
The two usual causes -- an activation no scalar scheme can stabilize, and a model whose
graph could not be fully observed -- are both knowable in advance, so both are reported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .core.stability import stable_depth

_RULE = "─"


@dataclass(frozen=True)
class LayerRecord:
    """One scalable layer: what it was given and, if measured, what came out."""

    name: str
    kind: str
    activation: str | None = None
    fan: float | None = None
    scale: float | None = None
    is_gain: bool = False
    predicted: float | None = None
    measured: float | None = None
    measured_count: int | None = None
    units: int | None = None
    constrained: bool = True
    """Whether the solve had a target here.

    A layer feeding a normalization has nothing downstream to aim at, since the
    normalization discards its scale, so its predicted level is not held against the
    prediction's accuracy.
    """
    fixed: bool = False
    """Whether the scale came from a user-fixed gain rather than from the solve."""
    notes: tuple[str, ...] = ()

    @property
    def deviation(self) -> float | None:
        """Relative gap between prediction and measurement, when both exist."""
        if self.predicted is None or self.measured is None or self.predicted <= 0.0:
            return None
        return abs(self.measured - self.predicted) / self.predicted

    @property
    def noise_floor(self) -> float | None:
        """Gap size that sampling alone explains.

        Two terms.  A measured second moment is an average of squares, so batch sampling
        contributes about ``sqrt(2/n)``.  The usually larger term is that an analytic
        prediction is an ensemble expectation while the model holds one draw, whose layer
        statistics deviate by about ``1/sqrt(units)`` however much data is pushed through.
        """
        if not self.measured_count:
            return None
        batch_term = 2.0 / max(self.measured_count, 1)
        draw_term = 1.0 / max(self.units or self.measured_count, 1)
        return 2.0 * math.sqrt(batch_term + draw_term)

    @property
    def excess_deviation(self) -> float | None:
        """How far the gap exceeds what sampling explains.  Zero means consistent."""
        if not self.constrained:
            return None
        gap, floor = self.deviation, self.noise_floor
        if gap is None:
            return None
        if floor is None:
            return gap
        return max(gap - floor, 0.0)


@dataclass(frozen=True)
class StabilityRecord:
    """Depth behavior of one activation."""

    activation: str
    chi: float
    sigma_star: float | None
    homogeneous_degree: float | None
    feasible_m2: tuple[float, float]
    verdict: str
    depth: int = 1

    @property
    def drift(self) -> float:
        """How far a unit relative error grows over the observed depth."""
        try:
            return float(self.chi**self.depth)
        except OverflowError:  # pragma: no cover
            return math.inf

    def advice(self) -> str | None:
        """Plain-language consequence, when the verdict has one."""
        if self.verdict == "unstable":
            degree = self.homogeneous_degree
            lead = (
                f"{self.activation} is homogeneous of degree {degree:g}"
                if degree is not None
                else f"{self.activation} expands variance"
            )
            return (
                f"{lead} (chi={self.chi:.3f}), so a relative error grows by "
                f"{self.chi:.2f}x per layer and reaches {self.drift:.3g}x over {self.depth} "
                "layers. No scalar initialization is depth-stable here: reduce depth, insert "
                "normalization, or use a degree-one activation"
            )
        if self.verdict == "expansive":
            return (
                f"{self.activation} expands variance (chi={self.chi:.3f}): a relative error "
                f"grows {self.chi:.2f}x per layer, {self.drift:.3g}x over these {self.depth} "
                f"layers. Usable at this depth; past about {stable_depth(self.chi):.0f} layers "
                "it would not be"
            )
        if self.verdict == "infeasible":
            return (
                f"{self.activation} saturates (E[a^2] <= {self.feasible_m2[1]:.4g}), so forward "
                "variance cannot be preserved; AnyInit targeted the middle of its reachable "
                "variance range instead"
            )
        return None


@dataclass(frozen=True)
class InitReport:
    """Result of one ``initialize`` call."""

    backend: str
    mode: str
    graph_fidelity: str
    config: dict[str, Any] = field(default_factory=dict)
    layers: tuple[LayerRecord, ...] = ()
    stability: tuple[StabilityRecord, ...] = ()
    warnings: tuple[str, ...] = ()
    iterations: int = 0
    converged: bool = True
    objective_error: float = 0.0
    """Largest relative gap between an objective and what the solve achieved."""
    params: Any = None
    """New parameter tree, for functional backends such as JAX.  ``None`` elsewhere."""
    unscaled: tuple[tuple[str, str], ...] = ()
    """Weights left as the framework initialized them, each with the reason."""

    # ------------------------------------------------------------- inspection

    @property
    def max_deviation(self) -> float | None:
        """Worst prediction-vs-measurement gap beyond sampling noise."""
        gaps = [r.excess_deviation for r in self.layers if r.excess_deviation is not None]
        return max(gaps) if gaps else None

    @property
    def worst_layer(self) -> LayerRecord | None:
        """Layer with the worst unexplained prediction gap."""
        candidates = [r for r in self.layers if r.excess_deviation is not None]
        return max(candidates, key=lambda r: r.excess_deviation or 0.0) if candidates else None

    @property
    def unstable(self) -> tuple[StabilityRecord, ...]:
        """Activations that are not depth-stable or whose target is unreachable."""
        return tuple(s for s in self.stability if s.verdict in ("unstable", "infeasible"))

    def assert_healthy(self, tol: float = 0.15) -> None:
        """Raise when the run is not trustworthy.  Handy in tests and CI."""
        problems = []
        if not self.converged:
            problems.append("the solve did not converge")
        deviation = self.max_deviation
        if deviation is not None and deviation > tol:
            worst = self.worst_layer
            where = f" at {worst.name!r}" if worst else ""
            problems.append(
                f"prediction is off by {deviation:.1%}{where}, beyond sampling noise "
                f"(tolerance {tol:.0%})"
            )
        problems.extend(
            f"{record.activation} is not depth-stable (chi={record.chi:.3f})"
            for record in self.stability
            if record.verdict == "unstable"
        )
        if problems:
            raise AssertionError("AnyInit report is not healthy: " + "; ".join(problems))

    # ---------------------------------------------------------------- output

    def to_dict(self) -> dict[str, Any]:
        """Render the report as plain data, for logging."""
        return {
            "backend": self.backend,
            "mode": self.mode,
            "graph_fidelity": self.graph_fidelity,
            "config": dict(self.config),
            "converged": self.converged,
            "iterations": self.iterations,
            "max_deviation": self.max_deviation,
            "layers": [
                {
                    "name": r.name,
                    "kind": r.kind,
                    "activation": r.activation,
                    "fan": r.fan,
                    "scale": r.scale,
                    "is_gain": r.is_gain,
                    "predicted": r.predicted,
                    "measured": r.measured,
                    "fixed": r.fixed,
                    "notes": list(r.notes),
                }
                for r in self.layers
            ],
            "stability": [
                {
                    "activation": s.activation,
                    "chi": s.chi,
                    "sigma_star": s.sigma_star,
                    "homogeneous_degree": s.homogeneous_degree,
                    "feasible_m2": list(s.feasible_m2),
                    "verdict": s.verdict,
                    "depth": s.depth,
                }
                for s in self.stability
            ],
            "warnings": list(self.warnings),
            "unscaled": [{"name": n, "reason": r} for n, r in self.unscaled],
        }

    def __str__(self) -> str:
        return self._render(markdown=False)

    def to_markdown(self) -> str:
        """Render the report as a Markdown table."""
        return self._render(markdown=True)

    def _render(self, *, markdown: bool) -> str:
        cfg = self.config
        head = (
            f"AnyInit — backend={self.backend}, mode={self.mode}, "
            f"distribution={cfg.get('distribution', '?')}, center={cfg.get('center', '?')}, "
            f"graph={self.graph_fidelity}"
        )
        detail = (
            f"  solve: {self.iterations} pass(es), "
            f"largest objective gap {self.objective_error:.2e}"
            + ("" if self.converged else ", NOT converged")
        )
        bullet = "- " if markdown else "  "
        out = [f"## {head}" if markdown else head, detail, ""]
        out.extend(self._layer_table(markdown))

        if self.stability:
            out.extend(["", "### Stability" if markdown else "Stability"])
            for record in self.stability:
                degree = (
                    f" degree={record.homogeneous_degree:g}"
                    if record.homogeneous_degree is not None
                    else ""
                )
                star = "-" if record.sigma_star is None else f"{record.sigma_star:.4f}"
                out.append(
                    f"{bullet}{record.activation:<18} chi={record.chi:6.3f}  sigma*={star:>8}"
                    f"  {record.verdict}{degree}"
                )
                advice = record.advice()
                if advice:
                    out.append(f"{'  - ' if markdown else '      '}! {advice}")

        deviation = self.max_deviation
        if deviation is not None:
            worst = self.worst_layer
            where = f" at {worst.name}" if worst else ""
            verdict = "ok" if deviation <= 0.15 else "CHECK"
            raw = worst.deviation if worst else None
            detail = f"  (raw gap {raw:.2%}, the rest is sampling noise)" if raw else ""
            out.extend(
                [
                    "",
                    "### Validation" if markdown else "Validation",
                    f"{bullet}largest unexplained prediction/measurement gap: "
                    f"{deviation:.2%}{where}  [{verdict}]{detail}",
                ]
            )
            if verdict == "CHECK" and self.mode == "analytic":
                remedy = (
                    "mode='empirical' corrects it"
                    if cfg.get("center")
                    else "center=True removes most of it, mode='empirical' all of it"
                )
                out.append(
                    f"{bullet}the prediction is the ensemble average and this model is one "
                    f"draw of it, which drifts further from it with depth; {remedy}"
                )

        if self.unscaled:
            out.extend(["", "### Not scaled" if markdown else "Not scaled"])
            out.extend(f"  - {name}: {reason}" for name, reason in self.unscaled)

        if self.warnings:
            out.extend(["", "### Warnings" if markdown else "Warnings"])
            out.extend(f"  - {w}" for w in self.warnings)

        return "\n".join(out)

    def _layer_table(self, markdown: bool) -> Sequence[str]:
        header = ("layer", "kind", "activation", "fan", "scale", "pred", "meas")
        rows = [
            (
                r.name,
                r.kind,
                (r.activation or "-") + (" (fixed gain)" if r.fixed else ""),
                "-" if r.fan is None else f"{r.fan:g}",
                "-" if r.scale is None else f"{'gain=' if r.is_gain else ''}{r.scale:.6f}",
                "-" if r.predicted is None else f"{r.predicted:.4f}",
                "-" if r.measured is None else f"{r.measured:.4f}",
            )
            for r in self.layers
        ]
        if markdown:
            return [
                "| " + " | ".join(header) + " |",
                "|" + "---|" * len(header),
                *("| " + " | ".join(row) + " |" for row in rows),
            ]

        widths = [
            max(len(header[i]), *(len(row[i]) for row in rows)) if rows else len(header[i])
            for i in range(len(header))
        ]

        def render(row: Sequence[str]) -> str:
            # Names left-aligned, numbers right-aligned.
            cells = (
                cell.ljust(widths[i]) if i < 3 else cell.rjust(widths[i])
                for i, cell in enumerate(row)
            )
            return "  " + "  ".join(cells)

        return [
            render(header),
            "  " + _RULE * (sum(widths) + 2 * len(widths)),
            *(render(r) for r in rows),
        ]
