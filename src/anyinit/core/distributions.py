"""Weight samplers, all normalized to unit variance.

The solver decides a layer's scale; a sampler only produces the shape of the distribution.
Each divides out its own realized standard deviation, so applying a scale afterward is an
exact multiplication rather than an inherited approximation.

Sampling happens in NumPy, in canonical layout, so the same seed yields the same weights
under PyTorch, TensorFlow and JAX.
"""

from __future__ import annotations

import math

import numpy as np

from .fan import ParamSpec

#: Available distributions.
NORMAL = "normal"
UNIFORM = "uniform"
SINUSOIDAL = "sinusoidal"

DISTRIBUTIONS: tuple[str, ...] = (NORMAL, UNIFORM, SINUSOIDAL)

#: Distributions whose rows are not independent, so the solver's i.i.d. assumption only
#: holds approximately.  Flagged in the report rather than silently accepted.
NON_IID = frozenset({SINUSOIDAL})

#: Distributions whose rows already sum to zero, so the layer discards its input's mean
#: whether or not centering was asked for.  The recursion has to use its centered form for
#: these: a sinusoid summed over a whole period is zero by construction.
INHERENTLY_CENTERED = frozenset({SINUSOIDAL})


def sample(
    spec: ParamSpec,
    distribution: str,
    rng: np.random.Generator,
    *,
    center: bool = False,
) -> np.ndarray:
    """Draw a unit-variance weight array in canonical layout."""
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"unknown distribution {distribution!r}; expected one of {DISTRIBUTIONS}")
    shape = spec.canonical_shape

    if distribution == NORMAL:
        w = rng.standard_normal(shape)
    elif distribution == UNIFORM:
        bound = math.sqrt(3.0)  # Var[U(-sqrt3, sqrt3)] == 1
        w = rng.uniform(-bound, bound, size=shape)
    else:
        w = _sinusoidal(shape, rng)

    if center and can_center(spec):
        # Remove each output unit's mean so the unit contributes no DC component.
        w = w - w.mean(axis=tuple(range(1, w.ndim)), keepdims=True)

    return _normalize(w)


#: Centering costs the layer one input direction, so it only applies above this fan_in.
MIN_CENTERING_FAN = 8


def is_centered(distribution: str, spec: ParamSpec, requested: bool) -> bool:
    """Whether a layer's draw will have rows summing to zero."""
    if distribution in INHERENTLY_CENTERED:
        return True
    return requested and can_center(spec)


def can_center(spec: ParamSpec) -> bool:
    """Whether a layer is wide enough to be centered.

    Centering constrains every row to ``sum_i W_ji == 0``, so the layer loses one input
    direction: the all-ones one, along which its input's mean lies.  Negligible for a wide
    layer, fatal for a narrow one -- at ``fan_in == 2`` the rank drops to one.
    """
    shape = spec.canonical_shape
    if len(shape) < 2:
        return False
    inputs = 1
    for dim in shape[1:]:
        inputs *= int(dim)
    return inputs >= MIN_CENTERING_FAN


def _normalize(w: np.ndarray) -> np.ndarray:
    """Rescale to exactly unit standard deviation, or to unit magnitude if degenerate."""
    if w.size < 2:
        return np.ones_like(w)
    std = float(w.std())
    if not math.isfinite(std) or std <= 0.0:
        return np.ones_like(w)
    return w / std


def _sinusoidal(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Structured sinusoidal weights, one distinct wave per output unit.

    Frequencies stay strictly below Nyquist and phases span ``[0, pi)``, spread evenly
    among the units sharing a frequency.  All three bounds are needed to keep the rows
    independent: frequencies at or past Nyquist alias onto each other, and phases a half
    turn apart differ only in sign.

    Described in Fernandez-Hernandez et al., *Sinusoidal Initialization* (2025).
    """
    n_out = shape[0]
    n_in = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    if n_in < 2:
        return rng.standard_normal(shape)

    n_freq = max(1, (n_in - 1) // 2)
    unit = np.arange(n_out)
    freq_index = (unit % n_freq) + 1
    slots = max(1, -(-n_out // n_freq))  # ceil: units sharing one frequency
    phase = math.pi * (unit // n_freq) / slots

    omega = 2.0 * math.pi * freq_index / n_in
    j = np.arange(n_in)
    w = np.sin(omega[:, None] * j[None, :] + phase[:, None])
    return w.reshape(shape)
