"""Whether an activation can hold a signal steady across depth.

An initialization sets one scalar per layer, which suffices only if the activation's depth
map is non-expansive.  With ``chi = d log E[f^2] / d log Var[z]`` above one, an error in
the variance is amplified by ``chi`` per layer and no choice of scalars survives depth.
For a positively homogeneous activation of degree ``p``, ``chi == p`` exactly: ReLU sits at
one, the edge of chaos, and ``relu**2`` and ``relu**3`` at two and three.

Whether that amplification matters depends on depth: ``chi**depth`` is how far a relative
error grows over the network.  So the verdict is *contractive* (``chi < 1``), *marginal*
(``chi`` at one), *expansive* (above one, but an error grows less than ``DRIFT_LIMIT``-fold
over the network's depth) or *unstable* (it grows more).  Without a depth, as for an
activation on its own, anything above one is *expansive*.

Bounded activations have the opposite problem: ``sup_sigma E[f^2]`` is finite, so a target
of one is unreachable and is detected rather than chased to the search bracket.  Those are
*infeasible*, and their ``chi`` is the slope of the output variance at the operating point
AnyInit substitutes, the middle of the reachable variance range.
"""

from __future__ import annotations

import math

import numpy as np

from .profile import ActivationProfile, Diagnostics

#: chi within this of one is marginal, the edge of chaos.
MARGINAL_BAND = 0.02
#: Largest growth of a relative error over the network's depth still called expansive.
DRIFT_LIMIT = 10.0

_SIGMA_MIN = 1e-6
_SIGMA_MAX = 1e4
_BISECTION_STEPS = 200
#: A target must sit this far below the reachable supremum to count as attainable.
_FEASIBILITY_MARGIN = 0.02


def m2_at(profile: ActivationProfile, sigma: float, mu: float = 0.0) -> float:
    """Second moment of the activation's output at a given input scale."""
    return profile.moments(mu, sigma * sigma).m2


def feasible_m2_range(profile: ActivationProfile, mu: float = 0.0) -> tuple[float, float]:
    """Range of second moments the activation can produce as ``sigma`` varies.

    The upper bound is ``inf`` for unbounded activations.  Detection compares two
    decades at the top of the scan: a bounded activation has flattened out by then.
    """
    grid = np.logspace(-3.0, 3.0, 61)
    values = [m2_at(profile, float(s), mu) for s in grid]
    low = min(values)
    high_near = m2_at(profile, 1e3, mu)
    high_far = m2_at(profile, 1e4, mu)
    if high_near <= 0.0:
        return (low, max(values))
    if high_far / high_near > 1.5:
        return (low, math.inf)
    return (low, max(*values, high_far))


def feasible_var_range(profile: ActivationProfile, mu: float = 0.0) -> tuple[float, float]:
    """Range of output *variances* the activation can produce as ``sigma`` varies.

    Distinct from :func:`feasible_m2_range` for activations with a DC offset: sigmoid's
    second moment never drops below 0.25 because its mean is 0.5, while its variance
    does start at zero.  Variance is the usable handle for those.
    """
    grid = np.logspace(-3.0, 3.0, 61)
    values = [profile.moments(mu, float(s) ** 2).var for s in grid]
    far = profile.moments(mu, 1e8).var
    near = profile.moments(mu, 1e6).var
    if near > 0.0 and far / near > 1.5:
        return (min(values), math.inf)
    return (min(values), max(*values, far))


def solve_input_std(profile: ActivationProfile, target_m2: float, mu: float = 0.0) -> float | None:
    """Bisect for ``sigma`` with ``E[f(N(mu, sigma^2))^2] == target_m2``.

    Returns ``None`` when the target is outside the activation's reachable range.
    Assumes ``E[f^2]`` is non-decreasing in ``sigma``, which holds for every activation
    in practical use.
    """
    if target_m2 <= 0.0:
        return None

    degree = profile.homogeneous_degree
    if degree is not None and degree > 0.0 and abs(mu) < 1e-12:
        base = profile.moments(0.0, 1.0).m2
        if base <= 0.0:
            return None
        return float((target_m2 / base) ** (1.0 / (2.0 * degree)))

    lo, hi = _SIGMA_MIN, _SIGMA_MAX
    # Require a strict bracket.  A bounded activation approaches its supremum
    # asymptotically, so a target at that supremum would otherwise "solve" at the edge
    # of the search range and report a saturating sigma as if it were a real answer.
    if m2_at(profile, hi, mu) <= target_m2 * (1.0 + _FEASIBILITY_MARGIN):
        return None
    if m2_at(profile, lo, mu) > target_m2:
        return None
    for _ in range(_BISECTION_STEPS):
        mid = math.sqrt(lo * hi)
        if m2_at(profile, mid, mu) < target_m2:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def lyapunov_slope(
    profile: ActivationProfile,
    sigma: float,
    mu: float = 0.0,
    *,
    eps: float = 1e-3,
    metric: str = "m2",
) -> float:
    """``d log E[f^2] / d log Var[z]`` by central difference in log-sigma.

    With ``metric="var"`` the output variance takes the place of ``E[f^2]``, which is the
    quantity a saturating activation's substituted objective holds.
    """
    degree = profile.homogeneous_degree
    if degree is not None and abs(mu) < 1e-12 and metric == "m2":
        return float(degree)  # Exact: E[f^2] scales as sigma**(2p), so the slope is p.

    def level(s: float) -> float:
        state = profile.moments(mu, s * s)
        return state.var if metric == "var" else state.m2

    hi = level(sigma * (1.0 + eps))
    lo = level(sigma * (1.0 - eps))
    if hi <= 0.0 or lo <= 0.0:
        return 0.0
    return (math.log(hi) - math.log(lo)) / (2.0 * math.log((1.0 + eps) / (1.0 - eps)))


def classify(chi: float, sigma_star: float | None, depth: int | None = None) -> str:
    """Turn a slope, an operating point and optionally a depth into a verdict."""
    if sigma_star is None:
        return "infeasible"
    if chi < 1.0 - MARGINAL_BAND:
        return "contractive"
    if chi <= 1.0 + MARGINAL_BAND:
        return "marginal"
    if depth is None or depth_error_factor(chi, depth) <= DRIFT_LIMIT:
        return "expansive"
    return "unstable"


def stable_depth(chi: float) -> float:
    """Layers over which a relative error grows by at most ``DRIFT_LIMIT``."""
    if chi <= 1.0 + MARGINAL_BAND:
        return math.inf
    return math.log(DRIFT_LIMIT) / math.log(chi)


def substituted_sigma(profile: ActivationProfile, mu: float = 0.0) -> float | None:
    """Input scale at which a bounded activation's variance is mid-range.

    That is the operating point AnyInit substitutes when a second-moment target is out of
    reach.  ``None`` when the variance range is unbounded.
    """
    low, high = feasible_var_range(profile, mu)
    if not math.isfinite(high):
        return None
    target = 0.5 * (low + high)
    lo, hi = _SIGMA_MIN, _SIGMA_MAX
    for _ in range(_BISECTION_STEPS):
        mid = math.sqrt(lo * hi)
        if profile.moments(mu, mid * mid).var < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def diagnose(profile: ActivationProfile, target_m2: float = 1.0) -> Diagnostics:
    """Full stability report for one activation, independent of any network's depth."""
    feasible = feasible_m2_range(profile)
    sigma_star = solve_input_std(profile, target_m2)
    if sigma_star is not None:
        operating = sigma_star
        chi = lyapunov_slope(profile, sigma_star)
    else:
        operating = substituted_sigma(profile) or 1.0
        chi = lyapunov_slope(profile, operating, metric="var")
    return Diagnostics(
        chi=chi,
        sigma_star=sigma_star,
        gain=operating,
        feasible_m2=feasible,
        feasible_var=feasible_var_range(profile),
        homogeneous_degree=profile.homogeneous_degree,
        verdict=classify(chi, sigma_star),
    )


def depth_error_factor(chi: float, depth: int) -> float:
    """How far a unit relative error drifts after ``depth`` layers."""
    try:
        return float(chi**depth)
    except OverflowError:  # pragma: no cover
        return math.inf
