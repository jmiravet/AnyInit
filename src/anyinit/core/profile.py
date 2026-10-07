"""The moment map of an activation.

Given a pre-activation ``z ~ N(mu, var)``, a profile returns the first, second and fourth
moments of ``f(z)``.  This is a map rather than a constant gain because, for anything but a
degree-one homogeneous activation, the answer depends on the variance that arrives.

Profiles hold a plain ``ndarray -> ndarray`` callable: the NumPy reference for builtin
activations, a backend-supplied bridge for user ones.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from . import quadrature as quad
from .moments import MomentState

EvalFn = Callable[[np.ndarray], np.ndarray]

#: Window used once per profile to locate kinks, wider than any quadrature window so one
#: detection pass serves every (mu, sigma) the solver asks about.
_KINK_WINDOW = 60.0


@dataclass(frozen=True)
class QuadratureConfig:
    """Knobs for the Gaussian integration behind a profile."""

    panels: int = quad.DEFAULT_PANELS
    nodes: int = quad.DEFAULT_NODES
    span: float = quad.DEFAULT_SPAN
    laguerre_nodes: int = 40

    def key(self) -> tuple[int, int, float, int]:
        """Hashable form, for cache keys."""
        return (self.panels, self.nodes, self.span, self.laguerre_nodes)


class ActivationProfile:
    """Moment map of a single activation, with its stability diagnostics."""

    def __init__(
        self,
        name: str,
        eval_fn: EvalFn,
        *,
        config: QuadratureConfig | None = None,
    ) -> None:
        self.name = name
        self._f = eval_fn
        self.config = config or QuadratureConfig()
        self._cache: dict[tuple[float, float], MomentState] = {}
        self._kinks: list[float] | None = None
        self._degree: float | None = None
        self._degree_probed = False
        self._base: MomentState | None = None
        self._diag: Diagnostics | None = None

    # ---------------------------------------------------------------- moments

    @property
    def kinks(self) -> list[float]:
        """Points where the activation loses smoothness, found once and reused."""
        if self._kinks is None:
            self._kinks = quad.detect_kinks(self._f, -_KINK_WINDOW, _KINK_WINDOW)
        return self._kinks

    def moments(self, mu: float = 0.0, var: float = 1.0) -> MomentState:
        """Moments of ``f(z)`` for ``z ~ N(mu, var)``."""
        if var < 0.0:
            raise ValueError(f"var must be non-negative, got {var}")
        if var == 0.0:
            value = float(np.asarray(self._f(np.array([mu]))).ravel()[0])
            return MomentState(value, value**2, value**4)

        key = (round(mu, 12), round(var, 12))
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        degree = self.homogeneous_degree
        if degree is not None and abs(mu) < 1e-12:
            # f(sigma * z) == sigma**p * f(z): one quadrature serves every scale.
            state = self._base_moments().scaled(math.sqrt(var) ** degree)
        else:
            state = self._quadrature_moments(mu, math.sqrt(var))
        self._cache[key] = state
        return state

    __call__ = moments

    def _quadrature_moments(self, mu: float, sigma: float) -> MomentState:
        cfg = self.config
        z, w = quad.gauss_legendre_nodes(
            mu, sigma, panels=cfg.panels, nodes=cfg.nodes, span=cfg.span, kinks=self.kinks
        )
        m1, m2, m4 = quad.integrate(self._f(z), w, (1, 2, 4))
        return MomentState(m1, m2, m4)

    def _base_moments(self) -> MomentState:
        if self._base is None:
            self._base = self._quadrature_moments(0.0, 1.0)
        return self._base

    def mixture_moments(self, m_s: float, v_s: float) -> MomentState:
        """Moments when the pre-activation is a Gamma scale mixture of Gaussians.

        See :func:`quadrature.scale_mixture_moments`.
        """
        if v_s <= 1e-14 * max(m_s * m_s, 1e-300):
            return self.moments(0.0, m_s)
        cfg = self.config
        m1, m2, m4 = quad.scale_mixture_moments(
            self._f,
            m_s,
            v_s,
            (1, 2, 4),
            laguerre_nodes=cfg.laguerre_nodes,
            panels=cfg.panels,
            nodes=cfg.nodes,
            span=cfg.span,
            kinks=self.kinks,
        )
        return MomentState(m1, m2, m4)

    def max_moments(self, mu: float, var: float, count: float) -> MomentState:
        """Moments of ``f(max of count i.i.d. N(mu, var))``.

        Max pooling after a monotone activation is pooling applied before it, since
        ``max_i f(z_i) == f(max_i z_i)``.  Integrating against the pre-activation maximum
        is therefore exact, where fitting a Gaussian to the activation's own moments is not
        -- a rectifier's output has half its mass at zero.
        """
        if count <= 1.0:
            return self.moments(mu, var)
        cfg = self.config
        z, w = quad.max_order_nodes(
            mu,
            math.sqrt(max(var, 1e-300)),
            count,
            panels=cfg.panels,
            nodes=cfg.nodes,
            span=cfg.span,
            kinks=self.kinks,
        )
        m1, m2, m4 = quad.integrate(self._f(z), w, (1, 2, 4))
        return MomentState(m1, m2, m4)

    def local_slope(self, mu: float, var: float, *, eps: float = 1e-3) -> float:
        """``d log E[f^2] / d log Var[z]`` at this operating point.

        :attr:`chi` describes the activation at its target; a solver stepping from
        elsewhere needs the slope where it stands, which for a saturating activation falls
        toward zero.
        """
        degree = self.homogeneous_degree
        if degree is not None and abs(mu) < 1e-12:
            return float(degree)  # Exact: E[f^2] scales as var**p.
        hi = self.moments(mu, var * (1.0 + eps)).m2
        lo = self.moments(mu, var * (1.0 - eps)).m2
        if hi <= 0.0 or lo <= 0.0:
            return 1.0
        slope = (math.log(hi) - math.log(lo)) / math.log((1.0 + eps) / (1.0 - eps))
        return slope if math.isfinite(slope) else 1.0

    # ------------------------------------------------------------- properties

    @property
    def homogeneous_degree(self) -> float | None:
        """Degree ``p`` if ``f(c*x) == c**p * f(x)``, else ``None``.

        A positive answer removes every quadrature call from the hot path, and the degree
        is the Lyapunov slope of the depth map.
        """
        if not self._degree_probed:
            self._degree = _probe_homogeneity(self._f)
            self._degree_probed = True
        return self._degree

    @property
    def diagnostics(self) -> Diagnostics:
        """Stability verdict for this activation, computed once."""
        if self._diag is None:
            from .stability import diagnose

            self._diag = diagnose(self)
        return self._diag

    @property
    def chi(self) -> float:
        """Lyapunov slope ``d log E[f^2] / d log Var[z]`` at the operating point."""
        return self.diagnostics.chi

    @property
    def gain(self) -> float:
        """Standard deviation of the pre-activation this activation is initialized to take.

        The classical gain: a layer scaled ``gain / sqrt(fan_in)`` and fed a unit second
        moment hands the activation exactly this.  ``sqrt(2)`` for ReLU.  For a bounded
        activation, whose ``E[f^2] = 1`` is out of reach, it is the input scale at the middle
        of its reachable variance range, which is what AnyInit aims at instead.
        """
        return self.diagnostics.gain

    @property
    def feasible_m2(self) -> tuple[float, float]:
        """Range of second moments the activation can produce."""
        return self.diagnostics.feasible_m2

    @property
    def feasible_var(self) -> tuple[float, float]:
        """Range of variances the activation can produce."""
        return self.diagnostics.feasible_var

    def solve_input_std(self, target_m2: float, mu: float = 0.0) -> float | None:
        """Smallest ``sigma`` with ``E[f(N(mu, sigma^2))^2] == target_m2``.

        ``None`` when the target is outside the activation's reachable range.
        """
        from .stability import solve_input_std

        return solve_input_std(self, target_m2, mu=mu)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ActivationProfile({self.name!r})"


# ------------------------------------------------------------------ probing


def _probe_homogeneity(
    f: EvalFn, *, tol: float = 1e-6, scales: Sequence[float] = (0.5, 2.0, 4.0)
) -> float | None:
    """Estimate the homogeneity degree, or return ``None`` if there is none."""
    x = np.concatenate([np.linspace(-4.0, -0.05, 97), np.linspace(0.05, 4.0, 97)])
    try:
        base = np.asarray(f(x), dtype=np.float64)
    except Exception:
        return None
    if base.shape != x.shape or not np.all(np.isfinite(base)):
        return None
    base_norm = float(np.linalg.norm(base))
    if base_norm < 1e-12:
        return None

    degrees = []
    for c in scales:
        scaled = np.asarray(f(c * x), dtype=np.float64)
        if scaled.shape != x.shape or not np.all(np.isfinite(scaled)):
            return None
        ratio = float(np.linalg.norm(scaled)) / base_norm
        if ratio <= 0.0:
            return None
        degrees.append(math.log(ratio) / math.log(c))

    degree = sum(degrees) / len(degrees)
    if max(abs(d - degree) for d in degrees) > 1e-4:
        return None

    # Confirm pointwise, not just in norm.
    for c in scales:
        predicted = (c**degree) * base
        actual = np.asarray(f(c * x), dtype=np.float64)
        if not np.allclose(actual, predicted, rtol=tol, atol=tol * max(base_norm, 1.0)):
            return None

    rounded = round(degree)
    return float(rounded) if abs(degree - rounded) < 1e-9 else float(degree)


@dataclass(frozen=True)
class Diagnostics:
    """What :mod:`anyinit.core.stability` concludes about an activation."""

    chi: float
    sigma_star: float | None
    gain: float
    feasible_m2: tuple[float, float]
    feasible_var: tuple[float, float]
    homogeneous_degree: float | None
    verdict: str
