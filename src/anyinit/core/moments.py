"""The state that AnyInit propagates through a network.

Three raw moments summarize a tensor's distribution.  ``m4`` is carried alongside the mean
and second moment so the solver can detect heavy tails; see ``scale_mixture_moments`` in
:mod:`anyinit.core.quadrature`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Second-moment kurtosis above which a signal is treated as heavy-tailed and the
#: Gaussian pre-activation assumption gets the scale-mixture correction.
HEAVY_TAIL_THRESHOLD = 3.5


@dataclass(frozen=True)
class MomentState:
    """Raw moments ``E[a]``, ``E[a^2]`` and ``E[a^4]`` of an activation tensor."""

    mean: float
    m2: float
    m4: float

    @classmethod
    def standard_normal(cls) -> MomentState:
        """Moments of a standard normal signal, the default network input."""
        return cls(mean=0.0, m2=1.0, m4=3.0)

    @classmethod
    def of_values(cls, values: Any) -> MomentState:
        """Moments of the entries of an array, or of a single number."""
        flat = np.asarray(values, dtype=np.float64).ravel()
        squares = flat * flat
        return cls(float(flat.mean()), float(squares.mean()), float((squares * squares).mean()))

    @classmethod
    def zero(cls) -> MomentState:
        """Return an all-zero state."""
        return cls(mean=0.0, m2=0.0, m4=0.0)

    @property
    def var(self) -> float:
        """Variance, ``E[a^2] - E[a]^2``, floored at zero."""
        return max(self.m2 - self.mean * self.mean, 0.0)

    @property
    def std(self) -> float:
        """Standard deviation."""
        return math.sqrt(self.var)

    @property
    def var_of_square(self) -> float:
        """``Var[a^2]``, the dispersion the scale-mixture correction is matched to."""
        return max(self.m4 - self.m2 * self.m2, 0.0)

    @property
    def kurtosis_of_square(self) -> float:
        """``E[a^4] / E[a^2]^2``.  Equals 3 for a zero-mean Gaussian."""
        return self.m4 / max(self.m2 * self.m2, 1e-300)

    @property
    def is_heavy_tailed(self) -> bool:
        """Whether the signal is leptokurtic enough to need the mixture correction."""
        return self.kurtosis_of_square > HEAVY_TAIL_THRESHOLD

    @property
    def is_finite(self) -> bool:
        """Whether every moment is finite."""
        return all(math.isfinite(v) for v in (self.mean, self.m2, self.m4))

    def scaled(self, factor: float) -> MomentState:
        """Moments of ``factor * a``."""
        f2 = factor * factor
        return MomentState(self.mean * factor, self.m2 * f2, self.m4 * f2 * f2)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MomentState(mean={self.mean:.6g}, m2={self.m2:.6g}, m4={self.m4:.6g})"
