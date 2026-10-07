"""How each kind of node transforms the moments of its input.

These are the recursions the analytic solver iterates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from . import quadrature
from .fan import ParamSpec, fan_in
from .moments import MomentState
from .profile import ActivationProfile


def through_linear(
    state_in: MomentState,
    spec: ParamSpec,
    sigma_w: float,
    *,
    centered: bool = False,
) -> MomentState:
    """Pre-activation moments of a dense or convolutional layer.

    ``Var[z] = fan * sw^2 * E[x^2]`` for i.i.d. zero-mean weights.  ``E[x^2]`` rather than
    ``Var[x]``: a nonzero activation mean contributes its square to the next layer's
    variance.  Biases start at zero, so the output is centered.  ``centered`` switches to
    ``Var[x]``, matching weights whose rows sum to zero and so annihilate the input's mean
    instead of propagating it.

    The fourth moment uses the Gaussian closure ``E[z^4] = 3 Var[z]^2``; it is only needed
    to judge how far from Gaussian the input was.
    """
    signal = state_in.var if centered else state_in.m2
    var = fan_in(spec) * sigma_w * sigma_w * signal
    return MomentState(0.0, var, 3.0 * var * var)


def through_embedding(sigma_w: float) -> MomentState:
    """Moments of an embedding lookup.

    A lookup is a product with a one-hot vector, so the output is a row of the table and
    its moments are the table's own.  There is no fan to divide by.
    """
    var = sigma_w * sigma_w
    return MomentState(0.0, var, 3.0 * var * var)


def through_activation(
    state_in: MomentState,
    profile: ActivationProfile,
    *,
    fan: float | None = None,
    sigma_w: float | None = None,
    centered: bool = False,
) -> MomentState:
    """Post-activation moments.

    A heavy-tailed input makes the pre-activation a scale mixture of Gaussians rather than
    a Gaussian, which distorts the high moments.  Given the layer's fan and scale, a Gamma
    matched to that mixture corrects for it.
    """
    if state_in.is_heavy_tailed and fan is not None and sigma_w is not None:
        m_s = fan * sigma_w * sigma_w * (state_in.var if centered else state_in.m2)
        v_s = fan * (sigma_w**4) * state_in.var_of_square
        if m_s > 0.0:
            return profile.mixture_moments(m_s, v_s)
    return profile.moments(state_in.mean, state_in.var)


def through_normalization(gamma: float, beta: float = 0.0) -> MomentState:
    """Moments after an affine normalization layer.

    Normalization discards the incoming scale, so the recursion restarts here and the gain
    is the knob that matters when a normalization follows a layer.
    """
    m2 = gamma * gamma + beta * beta
    return MomentState(beta, m2, 3.0 * gamma**4 + 6.0 * gamma * gamma * beta * beta + beta**4)


def through_merge(states: Sequence[MomentState], op: str = "add") -> MomentState:
    """Moments after summing, concatenating or multiplying branches.

    Branches are assumed independent, which a residual block only approximately is.
    """
    if not states:
        return MomentState.zero()
    if len(states) == 1:
        return states[0]

    if op in ("add", "sum", "add_n"):
        mean = sum(s.mean for s in states)
        m2 = sum(s.m2 for s in states)
        for i, a in enumerate(states):
            for b in states[i + 1 :]:
                m2 += 2.0 * a.mean * b.mean
        var = max(m2 - mean * mean, 0.0)
        return MomentState(mean, m2, 3.0 * var * var + 6.0 * var * mean * mean + mean**4)

    if op in ("mul", "multiply", "prod"):
        mean = math.prod(s.mean for s in states)
        m2 = math.prod(s.m2 for s in states)
        return MomentState(mean, m2, math.prod(s.m4 for s in states))

    if op in ("cat", "concat", "concatenate"):
        return through_concat(states)

    # Unknown merge: the safest stand-in is the widest branch, and the caller warns.
    return max(states, key=lambda s: s.m2)


def through_concat(
    states: Sequence[MomentState], sizes: Sequence[int] | None = None
) -> MomentState:
    """Element-count weighted average of the branches' moments."""
    if sizes is None:
        sizes = [1] * len(states)
    total = float(sum(sizes))
    if total <= 0.0:
        return MomentState.zero()
    mean = sum(s.mean * n for s, n in zip(states, sizes, strict=False)) / total
    m2 = sum(s.m2 * n for s, n in zip(states, sizes, strict=False)) / total
    m4 = sum(s.m4 * n for s, n in zip(states, sizes, strict=False)) / total
    return MomentState(mean, m2, m4)


def through_average_pool(state: MomentState, window: int, correlation: float = 0.5) -> MomentState:
    """Moments after average pooling.

    Averaging ``k`` independent values divides the variance by ``k``; spatial correlation
    makes the real reduction smaller, and ``correlation`` interpolates between the two.
    """
    if window <= 1:
        return state
    effective = 1.0 + (window - 1.0) * (1.0 - correlation)
    var = state.var / effective
    m2 = var + state.mean * state.mean
    return MomentState(state.mean, m2, 3.0 * var * var + 6.0 * var * state.mean**2 + state.mean**4)


def through_max_pool(state: MomentState, window: int, correlation: float = 0.0) -> MomentState:
    """Moments after max pooling, from the order statistic.

    The maximum of ``k`` i.i.d. normals has density ``k * phi(t) * Phi(t)**(k-1)``, which
    the same Gaussian quadrature integrates exactly.  ``correlation`` defaults to zero: a
    maximum is decided in the tail of its window, where neighbors correlate least.

    Assumes a Gaussian input.  When an activation precedes the pooling the solver uses
    :meth:`ActivationProfile.max_moments`, which is exact for that case.
    """
    if window <= 1:
        return state
    effective = max(1.0 + (window - 1.0) * (1.0 - correlation), 1.0)
    m1, m2 = quadrature.max_of_normals(state.mean, state.std, effective)
    var = max(m2 - m1 * m1, 0.0)
    return MomentState(m1, m2, 3.0 * var * var + 6.0 * var * m1 * m1 + m1**4)


def through_dropout(state: MomentState, p: float) -> MomentState:
    """Return the moments after inverted dropout, as it acts in training mode.

    Its train-time rescaling multiplies the second moment by ``1/(1-p)``.
    """
    if p <= 0.0:
        return state
    if p >= 1.0:
        return MomentState.zero()
    factor = 1.0 / (1.0 - p)
    return MomentState(state.mean, state.m2 * factor, state.m4 * factor * factor)
