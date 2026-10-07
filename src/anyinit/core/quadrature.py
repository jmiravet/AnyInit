"""Deterministic Gaussian integration.

Computes ``E[f(z)^k]`` for ``z ~ N(mu, sigma^2)`` and an arbitrary ``f`` by composite
Gauss-Legendre quadrature over the truncated Gaussian, which reaches machine precision in
~128 evaluations and handles kinked functions once :func:`detect_kinks` has placed the
panel boundaries.

Choosing the abscissas is separated from evaluating on them: callers take the abscissas,
apply the activation in whichever framework owns it, and pass the values to
:func:`integrate`.  That keeps this module free of any framework import.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

#: Half-width of the integration window, in standard deviations.  At 10 sigma the
#: truncated tail holds ~1.5e-23 of the mass, far below float64 resolution.
DEFAULT_SPAN = 10.0
DEFAULT_PANELS = 32
DEFAULT_NODES = 16

_erf = np.vectorize(math.erf, otypes=[np.float64])

EvalFn = Callable[[np.ndarray], np.ndarray]


def _panel_edges(lo: float, hi: float, panels: int, kinks: Sequence[float] | None) -> np.ndarray:
    """Uniform panel edges with any kinks inserted, so no panel straddles one."""
    edges = np.linspace(lo, hi, panels + 1)
    if kinks:
        inside = [k for k in kinks if lo < k < hi]
        if inside:
            edges = np.unique(np.concatenate([edges, np.asarray(inside, dtype=float)]))
            # Drop edges that collapse a panel to numerical width.
            keep = np.concatenate([[True], np.diff(edges) > 1e-12 * max(hi - lo, 1.0)])
            edges = edges[keep]
    return edges


def gauss_legendre_nodes(
    mu: float,
    sigma: float,
    *,
    panels: int = DEFAULT_PANELS,
    nodes: int = DEFAULT_NODES,
    span: float = DEFAULT_SPAN,
    kinks: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(abscissas, weights)`` with ``sum(weights * g(abscissas)) == E[g(z)]``.

    The weights already fold in the Gaussian density and are renormalized to sum to
    one, which makes the rule exact for constants despite the truncation.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    t, w = np.polynomial.legendre.leggauss(nodes)
    edges = _panel_edges(mu - span * sigma, mu + span * sigma, panels, kinks)
    lo, hi = edges[:-1, None], edges[1:, None]
    z = 0.5 * (hi - lo) * t[None, :] + 0.5 * (hi + lo)
    weights = 0.5 * (hi - lo) * w[None, :]
    density = np.exp(-0.5 * ((z - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
    z = z.ravel()
    weights = (weights * density).ravel()
    total = weights.sum()
    if total > 0.0:
        weights = weights / total
    return z, weights


def integrate(values: np.ndarray, weights: np.ndarray, powers: Sequence[int]) -> tuple[float, ...]:
    """Integrate ``values`` raised to each power against ``weights``."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.shape != weights.shape:
        raise ValueError(
            f"activation returned {values.shape[0]} values for {weights.shape[0]} abscissas; "
            "the function must be applied elementwise"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("activation produced non-finite values over the integration window")
    return tuple(float(np.sum(weights * values**p)) for p in powers)


def _refine_kink(f: EvalFn, lo: float, hi: float, rounds: int = 3, samples: int = 256) -> float:
    """Sharpen a bracketed kink location by repeated local resampling.

    Three rounds take a grid-resolution guess down to float64 noise, which the accuracy of
    the surrounding panels depends on.
    """
    for _ in range(rounds):
        x = np.linspace(lo, hi, samples)
        y = np.asarray(f(x), dtype=np.float64).ravel()
        if y.shape != x.shape or not np.all(np.isfinite(y)):
            break
        d2 = np.abs(np.diff(y, n=2))
        if not np.any(d2 > 0):
            break
        idx = int(np.argmax(d2)) + 1
        step = x[1] - x[0]
        lo, hi = float(x[idx] - step), float(x[idx] + step)
    return 0.5 * (lo + hi)


def detect_kinks(
    f: EvalFn,
    lo: float,
    hi: float,
    *,
    samples: int = 2048,
    max_kinks: int = 8,
    include: Sequence[float] = (0.0,),
) -> list[float]:
    """Locate points where ``f`` loses smoothness on ``[lo, hi]``.

    Candidates come from spikes in the discrete second difference, refined by
    :func:`_refine_kink`.  ``include`` is added unconditionally, since activations
    overwhelmingly break at the origin and some break there only in a high derivative.

    A spurious candidate only adds a panel boundary, so the test is permissive.
    """
    picked: list[float] = [float(v) for v in include if lo < v < hi]
    min_gap = (hi - lo) / 64.0

    x = np.linspace(lo, hi, samples)
    y = np.asarray(f(x), dtype=np.float64).ravel()
    if y.shape == x.shape and np.all(np.isfinite(y)):
        d2 = np.abs(np.diff(y, n=2))
        if np.any(d2 > 0):
            positive = d2[d2 > 0]
            threshold = max(float(np.median(positive)) * 50.0, float(d2.max()) * 0.25)
            candidates = np.flatnonzero(d2 >= threshold) + 1
            # Everything spiking means the function is rough, not kinked.
            if 0 < candidates.size <= samples // 8:
                step = float(x[1] - x[0])
                for idx in candidates[np.argsort(d2[candidates - 1])[::-1]]:
                    if len(picked) >= max_kinks:
                        break
                    value = _refine_kink(f, float(x[idx]) - step, float(x[idx]) + step)
                    if all(abs(value - q) > min_gap for q in picked):
                        picked.append(value)
    return sorted(picked)


def gamma_laguerre_nodes(
    mean: float, var: float, *, nodes: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature over ``S ~ Gamma`` matched to ``mean`` and ``var``.

    Used by :func:`scale_mixture_moments`.  Returns ``(scales, weights)`` with
    weights summing to one.
    """
    if var <= 0.0:
        return np.array([mean]), np.array([1.0])
    shape = mean * mean / var
    scale = var / mean
    t, w = np.polynomial.laguerre.laggauss(nodes)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        logw = np.log(w) + (shape - 1.0) * np.log(t)
    logw -= logw.max()
    weights = np.exp(logw)
    weights = weights / weights.sum()
    return scale * t, weights


def scale_mixture_moments(
    eval_fn: EvalFn,
    m_s: float,
    v_s: float,
    powers: Sequence[int],
    *,
    laguerre_nodes: int = 40,
    panels: int = DEFAULT_PANELS,
    nodes: int = DEFAULT_NODES,
    span: float = DEFAULT_SPAN,
    kinks: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Moments of ``f(z)`` where ``z | S ~ N(0, S)`` and ``S`` is Gamma distributed.

    A pre-activation is Gaussian only in the infinite-width limit.  At finite width it is a
    scale mixture -- ``z | a ~ N(0, sw^2 * sum a_i^2)`` -- whose dispersion inflates the
    high moments.  A Gamma matched to that dispersion corrects for it.
    """
    scales, sweights = gamma_laguerre_nodes(m_s, v_s, nodes=laguerre_nodes)
    totals = np.zeros(len(powers), dtype=np.float64)
    for s, sw in zip(scales, sweights, strict=False):
        sigma = math.sqrt(max(float(s), 1e-300))
        z, w = gauss_legendre_nodes(0.0, sigma, panels=panels, nodes=nodes, span=span, kinks=kinks)
        moments = integrate(eval_fn(z), w, powers)
        totals += sw * np.asarray(moments)
    return tuple(float(v) for v in totals)


def max_of_normals(mu: float, sigma: float, count: float) -> tuple[float, float]:
    """First two moments of the maximum of ``count`` i.i.d. ``N(mu, sigma^2)`` values.

    The maximum's density is ``k * phi(t) * Phi(t)**(k-1)``, so reweighting the standard
    rule by ``k * Phi(t)**(k-1)`` integrates it exactly, including for small ``k``.
    ``count`` may be fractional, which is how a correlation correction is expressed.
    """
    if count <= 1.0 or sigma <= 0.0:
        return mu, mu * mu + sigma * sigma
    t, w = gauss_legendre_nodes(0.0, 1.0)
    cdf = 0.5 * (1.0 + _erf(t / math.sqrt(2.0)))
    weights = w * count * np.power(np.clip(cdf, 1e-300, 1.0), count - 1.0)
    total = weights.sum()
    if total <= 0.0:
        return mu, mu * mu + sigma * sigma
    weights = weights / total
    values = mu + sigma * t
    return float(np.sum(weights * values)), float(np.sum(weights * values * values))


def max_order_nodes(
    mu: float,
    sigma: float,
    count: float,
    *,
    panels: int = DEFAULT_PANELS,
    nodes: int = DEFAULT_NODES,
    span: float = DEFAULT_SPAN,
    kinks: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Abscissas and weights for the maximum of ``count`` i.i.d. ``N(mu, sigma^2)``.

    Integrating an arbitrary function against these gives max pooling placed after a
    monotone activation, since ``max_i f(z_i) == f(max_i z_i)``.
    """
    z, w = gauss_legendre_nodes(mu, sigma, panels=panels, nodes=nodes, span=span, kinks=kinks)
    if count <= 1.0:
        return z, w
    cdf = 0.5 * (1.0 + _erf((z - mu) / (sigma * math.sqrt(2.0))))
    weights = w * count * np.power(np.clip(cdf, 1e-300, 1.0), count - 1.0)
    total = weights.sum()
    return z, (weights / total if total > 0.0 else w)
