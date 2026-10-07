"""Where activations are declared and their profiles cached.

Builtins are referenced by canonical name: a backend reports ``"gelu"`` and the registry
profiles the NumPy reference from :mod:`anyinit.core.activations`, so no framework is
involved and the numbers match across backends.

User activations keep their own framework.  ``register_activation`` stores the native
callable and the registry wraps it so quadrature abscissas round-trip through that
framework.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import ConfigError
from . import activations as builtin
from .activations import Array
from .profile import ActivationProfile

EvalFn = Callable[[Array], Array]


@dataclass(frozen=True)
class ActivationRef:
    """A reference to an activation as it appears in a graph.

    ``params`` carries values that change the function's shape -- a LeakyReLU slope, an ELU
    alpha -- so differently configured instances get different profiles.
    """

    name: str
    params: tuple[tuple[str, float], ...] = ()

    @classmethod
    def of(cls, name: str, **params: float) -> ActivationRef:
        """Build a reference from a name and keyword parameters."""
        return cls(name, tuple(sorted(params.items())))

    @property
    def kwargs(self) -> dict[str, float]:
        """Shape parameters as a plain dict."""
        return dict(self.params)

    def __str__(self) -> str:
        if not self.params:
            return self.name
        inner = ", ".join(f"{k}={v:g}" for k, v in self.params)
        return f"{self.name}({inner})"


@dataclass(frozen=True)
class ActivationSpec:
    """How to build a profile for one registered activation."""

    name: str
    numpy_fn: EvalFn | None = None
    native_fn: Any = None
    native_type: Any = None

    @property
    def needs_backend(self) -> bool:
        """Whether profiling this activation requires a framework."""
        return self.numpy_fn is None


class ActivationRegistry:
    """Process-wide registry of activations and their cached profiles."""

    def __init__(self) -> None:
        self._specs: dict[str, ActivationSpec] = {}
        self._profiles: dict[tuple[str, tuple[tuple[str, float], ...], str], ActivationProfile] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------ registration

    def register(
        self,
        name: str,
        *,
        numpy_fn: EvalFn | None = None,
        native_fn: Any = None,
        native_type: Any = None,
        overwrite: bool = False,
    ) -> ActivationSpec:
        """Record an activation and discard any profile cached for its name."""
        if numpy_fn is None and native_fn is None:
            raise ConfigError("register needs either numpy_fn or native_fn")
        with self._lock:
            if name in self._specs and not overwrite:
                raise ConfigError(
                    f"activation {name!r} is already registered; pass overwrite=True to replace it"
                )
            spec = ActivationSpec(
                name=name,
                numpy_fn=numpy_fn,
                native_fn=native_fn,
                native_type=native_type,
            )
            self._specs[name] = spec
            # A re-registration must not leave stale profiles behind.
            for key in [k for k in self._profiles if k[0] == name]:
                del self._profiles[key]
            return spec

    def unregister(self, name: str) -> None:
        """Forget an activation and its cached profiles."""
        with self._lock:
            self._specs.pop(name, None)
            for key in [k for k in self._profiles if k[0] == name]:
                del self._profiles[key]

    def __contains__(self, name: object) -> bool:
        return name in self._specs or name in builtin.BUILTIN

    def spec(self, name: str) -> ActivationSpec | None:
        """Registration for a name, or ``None``."""
        return self._specs.get(name)

    @property
    def custom_names(self) -> tuple[str, ...]:
        """Names of the user-registered activations."""
        return tuple(self._specs)

    def native_types(self) -> dict[Any, str]:
        """Native classes a backend should treat as atomic activation nodes."""
        return {s.native_type: s.name for s in self._specs.values() if s.native_type is not None}

    def native_callables(self) -> dict[Any, str]:
        """Native functions a backend should treat as atomic activation nodes.

        Class registrations are excluded: their callable is an instance made only for
        profiling, and tracers recognize them through :meth:`native_types` instead.
        """
        return {
            s.native_fn: s.name
            for s in self._specs.values()
            if s.native_fn is not None and s.native_type is None and callable(s.native_fn)
        }

    # ----------------------------------------------------------------- lookup

    def profile(self, ref: ActivationRef, backend: Any = None) -> ActivationProfile | None:
        """Profile for ``ref``, built on first use and cached.

        ``None`` for unknown names and for non-pointwise ops such as softmax, whose
        moments are not a one-dimensional Gaussian integral.
        """
        if ref.name in builtin.NON_POINTWISE:
            return None
        tag = getattr(backend, "name", "numpy") or "numpy"
        key = (ref.name, ref.params, tag)
        with self._lock:
            hit = self._profiles.get(key)
            if hit is not None:
                return hit
            built = self._build(ref, backend)
            if built is not None:
                self._profiles[key] = built
            return built

    def _build(self, ref: ActivationRef, backend: Any) -> ActivationProfile | None:
        spec = self._specs.get(ref.name)
        if spec is not None:
            return self._from_spec(spec, ref, backend)

        fn = builtin.parameterized(ref.name, **ref.kwargs)
        if fn is None:
            return None
        return ActivationProfile(str(ref), fn)

    def _from_spec(
        self, spec: ActivationSpec, ref: ActivationRef, backend: Any
    ) -> ActivationProfile | None:
        if spec.numpy_fn is not None:
            fn: EvalFn = spec.numpy_fn
        else:
            if backend is None:
                raise ConfigError(
                    f"activation {spec.name!r} was registered as a native callable, so it needs "
                    "an active backend to evaluate; register a NumPy equivalent to profile it "
                    "without one"
                )
            native = spec.native_fn
            active = backend

            def fn(x: Array) -> Array:
                return np.asarray(active.eval_elementwise(native, x))

        if ref.kwargs:
            base, bound_kwargs = fn, ref.kwargs

            def fn(x: Array) -> Array:
                return base(x, **bound_kwargs)

        return ActivationProfile(str(ref), fn)


#: The registry every entry point uses.
REGISTRY = ActivationRegistry()
