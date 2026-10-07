"""NumPy reference implementations of the standard activations.

A backend reports that a node is, say, a ``gelu``; the core looks the name up here and
profiles the NumPy version.  Same closed forms, so the numbers match what the framework
would produce, and the core stays importable with nothing but NumPy.

User-defined activations stay in their own framework instead; see
:mod:`anyinit.core.registry`.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

#: Float array, spelled precisely so NumPy's ufunc overloads resolve under strict typing.
Array = npt.NDArray[np.floating]

ActivationFn = Callable[[Array], Array]

_SQRT_2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_erf = np.vectorize(math.erf, otypes=[np.float64])


def identity(x: Array) -> Array:
    """Return the input unchanged."""
    return x


def relu(x: Array) -> Array:
    """Rectified linear unit, ``max(x, 0)``."""
    return np.maximum(x, 0.0)


def relu6(x: Array) -> Array:
    """ReLU clipped above at six."""
    return np.clip(x, 0.0, 6.0)


def leaky_relu(x: Array, negative_slope: float = 0.01) -> Array:
    """ReLU with a finite slope on the negative side."""
    return np.where(x >= 0.0, x, negative_slope * x)


def elu(x: Array, alpha: float = 1.0) -> Array:
    """Exponential linear unit: linear above zero, ``alpha*(exp(x)-1)`` below."""
    return np.where(x > 0.0, x, alpha * (np.exp(np.minimum(x, 0.0)) - 1.0))


def selu(x: Array) -> Array:
    """Scaled ELU, whose constants make unit variance a fixed point."""
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * np.where(x > 0.0, x, alpha * (np.exp(np.minimum(x, 0.0)) - 1.0))


def celu(x: Array, alpha: float = 1.0) -> Array:
    """ELU with its negative branch scaled by ``alpha`` as well."""
    return np.where(x > 0.0, x, alpha * (np.exp(np.minimum(x, 0.0) / alpha) - 1.0))


def gelu(x: Array) -> Array:
    """Exact GELU, ``x * Phi(x)``."""
    return np.asarray(x * 0.5 * (1.0 + _erf(x / _SQRT_2)))


def gelu_tanh(x: Array) -> Array:
    """Apply the tanh approximation of GELU that several frameworks default to."""
    inner = _SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def sigmoid(x: Array) -> Array:
    """Logistic function, ``1/(1+exp(-x))``."""
    # Overflow-free form of 1/(1+exp(-x)).
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * x)))


def silu(x: Array) -> Array:
    """Sigmoid linear unit, ``x*sigmoid(x)``.  Also called swish."""
    return x * sigmoid(x)


def mish(x: Array) -> Array:
    """``x*tanh(softplus(x))``."""
    return x * np.tanh(softplus(x))


def softplus(x: Array, beta: float = 1.0) -> Array:
    """Smooth approximation of ReLU, ``log(1+exp(beta*x))/beta``."""
    bx = beta * x
    return np.where(bx > 20.0, x, np.log1p(np.exp(np.minimum(bx, 20.0))) / beta)


def softsign(x: Array) -> Array:
    """``x/(1+|x|)``."""
    return x / (1.0 + np.abs(x))


def tanh(x: Array) -> Array:
    """Hyperbolic tangent."""
    return np.tanh(x)


def hardtanh(x: Array, lo: float = -1.0, hi: float = 1.0) -> Array:
    """Identity clipped to ``[lo, hi]``."""
    return np.clip(x, lo, hi)


def hardsigmoid(x: Array) -> Array:
    """Piecewise-linear approximation of the logistic function."""
    return np.asarray(np.clip(x / 6.0 + 0.5, 0.0, 1.0))


def hardswish(x: Array) -> Array:
    """``x*hardsigmoid(x)``."""
    return x * hardsigmoid(x)


def exponential(x: Array) -> Array:
    """``exp(x)``, clamped to keep the result finite."""
    return np.exp(np.minimum(x, 60.0))


#: Canonical activation name -> NumPy implementation.  Backends translate their native ops
#: into these names; anything missing is reported as an unknown op.
BUILTIN: dict[str, ActivationFn] = {
    "identity": identity,
    "linear": identity,
    "relu": relu,
    "relu6": relu6,
    "leaky_relu": leaky_relu,
    "elu": elu,
    "selu": selu,
    "celu": celu,
    "gelu": gelu,
    "gelu_tanh": gelu_tanh,
    "sigmoid": sigmoid,
    "silu": silu,
    "swish": silu,
    "mish": mish,
    "softplus": softplus,
    "softsign": softsign,
    "tanh": tanh,
    "hardtanh": hardtanh,
    "hardsigmoid": hardsigmoid,
    "hardswish": hardswish,
    "exponential": exponential,
}

#: Activations whose output is not a pointwise function of the input, so one-dimensional
#: quadrature cannot profile them.  Treated as shape-preserving and reported.
NON_POINTWISE = frozenset({"softmax", "log_softmax", "glu", "logsigmoid_pair"})


def parameterized(name: str, **kwargs: float) -> ActivationFn | None:
    """Bind keyword parameters of a builtin, e.g. ``leaky_relu`` with its slope."""
    base = BUILTIN.get(name)
    if base is None:
        return None
    if not kwargs:
        return base

    def bound(x: Array) -> Array:
        return base(x, **kwargs)

    bound.__name__ = name
    return bound
