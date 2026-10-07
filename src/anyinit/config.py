"""Validated settings for one ``initialize`` call.

Everything is checked before a single weight is touched, so a bad argument raises rather
than leaving a half-initialized network behind.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .core.distributions import DISTRIBUTIONS, NON_IID
from .errors import ConfigError

MODES = ("analytic", "empirical")


@dataclass(frozen=True)
class InitConfig:
    """Resolved options for an initialization run."""

    mode: str = "analytic"
    input_spec: Any = None
    distribution: str = "normal"
    center: bool = False
    gains: Mapping[str, float] = field(default_factory=dict)
    """Activation name -> fixed gain.  Layers feeding that activation get
    ``gain / sqrt(fan_in)`` (or ``gain`` itself on a normalization) instead of a solved scale."""
    seed: int | None = None
    notes: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        mode: str = "analytic",
        input_spec: Any = None,
        *,
        distribution: str = "normal",
        center: bool = False,
        gains: Mapping[str, float] | None = None,
        seed: int | None = None,
    ) -> InitConfig:
        """Construct a config, rejecting bad values."""
        _one_of("mode", mode, MODES)
        _one_of("distribution", distribution, DISTRIBUTIONS)
        if mode == "empirical" and input_spec is None:
            raise ConfigError(
                "mode='empirical' needs data: pass input_spec as a batch, or a callable "
                "returning one. Use mode='analytic' to initialize without running the model"
            )
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ConfigError(f"seed must be an int or None, got {seed!r}")

        fixed: dict[str, float] = {}
        for name, value in dict(gains or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"gain for {name!r} must be a number, got {value!r}")
            if not math.isfinite(value) or value <= 0.0:
                raise ConfigError(f"gain for {name!r} must be positive and finite, got {value}")
            fixed[str(name)] = float(value)

        notes: tuple[str, ...] = ()
        if distribution in NON_IID:
            notes = (
                f"distribution {distribution!r} has correlated rows, which the analytic "
                "solver's i.i.d. recursion only approximates; use mode='empirical' to "
                "measure instead",
            )
        return cls(mode, input_spec, distribution, bool(center), fixed, seed, notes)

    def summary(self) -> dict[str, Any]:
        """Resolved options as plain data, for the report."""
        return {
            "mode": self.mode,
            "distribution": self.distribution,
            "center": self.center,
            "gains": dict(self.gains),
            "seed": self.seed,
        }


def _one_of(field_name: str, value: Any, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ConfigError(f"{field_name} must be one of {allowed}, got {value!r}")
