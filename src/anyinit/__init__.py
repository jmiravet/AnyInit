"""AnyInit: initialize any model, in any framework, correctly.

One call traces the architecture, works out which activation follows each layer, and
scales every weight so the signal neither dies nor explodes with depth::

    import anyinit

    report = anyinit.initialize(model)                       # data-free
    report = anyinit.initialize(model, "empirical", batch)    # measured
    print(report)

``model`` may be a ``torch.nn.Module``, a Keras model or a Flax module; the framework is
detected from it, and only the one in use needs to be installed.

An activation AnyInit has never seen is a first-class input.  Register it and its moment
map is measured by Gaussian quadrature, then used in either mode::

    @anyinit.register_activation
    def relu3(x):
        return torch.relu(x) ** 3

Some activations cannot be stabilized across depth by any initialization, and the report
says so rather than returning a dead network: ``relu3`` above is homogeneous of degree
three, so a relative error in the variance triples at every layer.  See
``report.stability``, or call ``report.assert_healthy()`` to raise on the finding.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from . import backends as _backends
from .config import InitConfig
from .core.activations import BUILTIN as BUILTIN_ACTIVATIONS
from .core.profile import ActivationProfile
from .core.registry import REGISTRY, ActivationRef
from .core.stability import depth_error_factor
from .errors import (
    AnyInitError,
    BackendNotFoundError,
    BackendUnavailableError,
    ConfigError,
    TraceError,
)
from .report import InitReport, LayerRecord, StabilityRecord

__version__ = "0.1.0"

__all__ = [
    "BUILTIN_ACTIVATIONS",
    "ActivationProfile",
    "AnyInitError",
    "BackendNotFoundError",
    "BackendUnavailableError",
    "ConfigError",
    "InitReport",
    "LayerRecord",
    "StabilityRecord",
    "TraceError",
    "__version__",
    "activation_profile",
    "available_backends",
    "depth_error_factor",
    "gain",
    "initialize",
    "initialize_params",
    "known_backends",
    "register_activation",
    "registered_activations",
    "unregister_activation",
]


def initialize(
    model: Any,
    mode: str = "analytic",
    input_spec: Any = None,
    *,
    distribution: str = "normal",
    center: bool = False,
    gains: Mapping[str, float] | None = None,
    seed: int | None = None,
    params: Any = None,
) -> InitReport:
    """Reinitialize ``model`` so its activation statistics hold across depth.

    Args:
        model: A ``torch.nn.Module``, a Keras model or layer, or a Flax module.
        mode: ``"analytic"`` propagates moments through the graph without running the
            model; ``"empirical"`` measures real batches and assumes nothing.
        input_spec: A shape, a batch, or a callable returning a batch.  Optional in
            analytic mode, where it is used only for the validation pass; required in
            empirical mode and for Flax.
        distribution: Shape of the weight draw: ``"normal"``, ``"uniform"`` or
            ``"sinusoidal"``.
        center: Remove each output unit's weight mean, so that a layer discards the mean
            of its input instead of passing it on.
        gains: Activation name -> fixed gain, e.g. ``{"relu": 2**0.5}``.  Layers feeding
            that activation get ``gain / sqrt(fan_in)`` instead of a solved scale.
        seed: Seed for the weight draw.  The same seed gives the same weights in every
            framework.
        params: Parameter tree for functional frameworks.  The new tree comes back as
            ``report.params``.

    Returns:
        An :class:`~anyinit.report.InitReport`: the scale chosen for every layer, a
        depth-stability verdict per activation, and anything AnyInit could not do.

    Raises:
        ConfigError: An option is invalid.  Raised before the model is touched.
        BackendNotFoundError: The object is not a model of a supported framework.

    """
    config = InitConfig.build(
        mode, input_spec, distribution=distribution, center=center, gains=gains, seed=seed
    )
    from ._run import run

    return run(model, config, params=params)


def initialize_params(
    model: Any, params: Any, input_spec: Any = None, mode: str = "analytic", **options: Any
) -> tuple[Any, InitReport]:
    """Functional form for JAX and other immutable-parameter frameworks.

    As :func:`initialize`, but returns the new parameter tree alongside the report::

        params, report = anyinit.initialize_params(model, params, (1, 32))
    """
    report = initialize(model, mode, input_spec, params=params, **options)
    return report.params, report


def register_activation(
    activation: Any = None, *, name: str | None = None, overwrite: bool = False
) -> Any:
    """Teach AnyInit an activation it does not know.

    Usable bare, with arguments, or as a plain call::

        @anyinit.register_activation
        def relu3(x):
            return torch.relu(x) ** 3

        @anyinit.register_activation(name="relu_cubed")
        class ReLU3(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x) ** 3

        anyinit.register_activation(lambda x: np.maximum(x, 0) ** 3, name="relu3")

    The moment map is computed by Gaussian quadrature rather than looked up, and both
    modes then use it.  Registering also makes the activation visible to the tracers, so a
    module subclass becomes a single graph node instead of being inlined into primitives.

    A function written against NumPy works with every backend and needs no framework; one
    written against a specific framework is evaluated through it.  Which it is gets
    detected.

    Args:
        activation: A callable, or an ``nn.Module``/Keras layer subclass.
        name: Registry name.  Defaults to the function or class name, lowercased.
        overwrite: Permit replacing an existing registration of the same name.

    Returns:
        The activation itself, so this works as a decorator.

    """
    if activation is None:
        return lambda target: register_activation(target, name=name, overwrite=overwrite)

    callable_obj, native_type = _as_callable(activation)
    use_numpy = _is_numpy_callable(callable_obj)
    REGISTRY.register(
        name or _default_name(activation),
        numpy_fn=callable_obj if use_numpy else None,
        native_fn=None if use_numpy else callable_obj,
        native_type=native_type,
        overwrite=overwrite,
    )
    return activation


def unregister_activation(name: str) -> None:
    """Remove a registration and discard its cached profile."""
    REGISTRY.unregister(name)


def registered_activations() -> tuple[str, ...]:
    """Names of user-registered activations."""
    return REGISTRY.custom_names


def activation_profile(name: str, backend: Any = None, **params: float) -> ActivationProfile:
    """Profile of a registered or builtin activation, for inspection.

    ``activation_profile("relu3").chi`` says whether an activation can hold a signal across
    depth before anything is built with it.  ``backend`` is needed only for natively
    registered code, and is worked out from the registration when omitted.
    """
    reference = ActivationRef.of(name, **params)
    if backend is None:
        backend = _backend_for(name)
    profile = REGISTRY.profile(reference, backend)
    if profile is None:
        known = ", ".join(sorted(set(BUILTIN_ACTIVATIONS) | set(REGISTRY.custom_names)))
        raise ConfigError(f"no profile for activation {name!r}; known activations: {known}")
    return profile


def gain(name: str, **params: float) -> float:
    """Gain AnyInit gives an activation: weights scaled ``gain / sqrt(fan_in)``.

    The input standard deviation that lands ``E[f(z)^2]`` on one, so ``gain("relu")`` is
    ``sqrt(2)``.  A bounded activation cannot reach one; its gain is the input scale at the
    middle of its reachable variance range.  Parameters such as ``negative_slope`` go in
    as keywords.  Pass the result, or any other value, to ``initialize(gains=...)`` to fix
    it instead of solving for it.
    """
    return activation_profile(name, **params).gain


def available_backends() -> tuple[str, ...]:
    """Names of the backends whose framework is installed here.

    Checked without importing any framework.
    """
    return tuple(cls.name for cls in _backends.installed())


def known_backends() -> tuple[str, ...]:
    """Names of every backend AnyInit ships, installed or not."""
    return tuple(cls.name for cls in _backends.known())


# ------------------------------------------------------------------- helpers


def _backend_for(name: str) -> Any:
    """Backend able to evaluate a natively-registered activation, if one is needed."""
    spec = REGISTRY.spec(name)
    if spec is None or not spec.needs_backend:
        return None
    roots = _backends.framework_roots(spec.native_type or spec.native_fn)
    for cls in _backends.installed():
        if cls.frameworks and roots & set(cls.frameworks):
            return _instantiate(cls)
    return None


def _instantiate(cls: Any) -> Any:
    """Build a backend, returning ``None`` if its framework will not import."""
    try:
        return cls()
    except Exception:
        return None


def _default_name(activation: Any) -> str:
    if isinstance(activation, type):
        return activation.__name__.lower()
    explicit = getattr(activation, "__name__", None)
    if explicit and explicit != "<lambda>":
        return str(explicit)
    return type(activation).__name__.lower()


def _as_callable(activation: Any) -> tuple[Any, Any]:
    """Return ``(callable, native_type)``.

    A class is instantiated so it can be profiled, and kept so the tracers recognize its
    instances inside a model.
    """
    if isinstance(activation, type):
        return activation(), activation
    if not callable(activation):
        raise ConfigError(f"activation must be callable, got {type(activation).__name__}")
    native_type = type(activation) if _looks_like_layer(activation) else None
    return activation, native_type


def _looks_like_layer(activation: Any) -> bool:
    roots = _backends.module_roots(activation)
    return bool(roots & {"torch", "keras", "tensorflow", "flax"})


def _is_numpy_callable(fn: Any) -> bool:
    """Whether ``fn`` can be applied directly to a NumPy array."""
    probe = np.array([-1.0, 0.0, 1.0])
    try:
        out = fn(probe)
    except Exception:
        return False
    return isinstance(out, np.ndarray) and out.shape == probe.shape
