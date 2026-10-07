"""Backend discovery.

Adapter modules import no framework at module scope, so every backend is *known* whatever
is installed, and :func:`installed` reports which of them can actually run here.
Detection reads a model's class hierarchy rather than calling ``isinstance``; a model's
own framework is necessarily imported already, and no other one ever is.
"""

from __future__ import annotations

import importlib
from typing import Any

from ..errors import BackendNotFoundError, BackendUnavailableError
from .base import Backend, TapRecorder, framework_roots, module_roots

#: Known backends in detection order, most specific first.  Keras precedes PyTorch
#: because Keras 3 on its torch backend builds layers that subclass ``nn.Module``; the
#: PyTorch adapter also rejects Keras objects, so correctness does not rest on the order.
_BACKENDS: tuple[tuple[str, str], ...] = (
    ("anyinit.backends.keras", "KerasBackend"),
    ("anyinit.backends.jax", "FlaxBackend"),
    ("anyinit.backends.pytorch", "TorchBackend"),
)

_CLASS_CACHE: dict[str, type[Backend]] = {}


def known() -> list[type[Backend]]:
    """Every backend class, whether or not its framework is installed."""
    found: list[type[Backend]] = []
    for module_path, class_name in _BACKENDS:
        key = f"{module_path}.{class_name}"
        cached = _CLASS_CACHE.get(key)
        if cached is None:
            try:
                module = importlib.import_module(module_path)
            except ImportError:  # pragma: no cover - a backend module itself is broken
                continue
            cached = getattr(module, class_name)
            _CLASS_CACHE[key] = cached
        found.append(cached)
    return found


def installed() -> list[type[Backend]]:
    """Backend classes whose framework can be imported in this environment."""
    return [cls for cls in known() if cls.installed()]


def resolve(model: Any) -> Backend:
    """Pick and instantiate the backend for ``model``."""
    for cls in known():
        try:
            matched = cls.handles(model)
        except Exception:
            matched = False
        if not matched:
            continue
        try:
            return cls()
        except ImportError as exc:
            raise BackendUnavailableError(
                f"the {cls.name} backend recognized this model but {cls.requires!r} could not "
                f"be imported: {exc}. Reinstall it with: pip install {cls.install}"
            ) from exc

    raise BackendNotFoundError(_not_found_message(model))


def _not_found_message(model: Any) -> str:
    kind = f"{type(model).__module__}.{type(model).__qualname__}"
    ready = installed()
    if not ready:
        return (
            f"no AnyInit backend recognizes {kind}, and no supported framework is installed. "
            "AnyInit works with PyTorch, TensorFlow (Keras 3) and JAX (Flax); install the one "
            "your model uses and it will be picked up automatically"
        )
    names = ", ".join(cls.name for cls in ready)
    return (
        f"no AnyInit backend recognizes {kind}. Installed backends: {names}. "
        "PyTorch models must subclass torch.nn.Module, TensorFlow models must be Keras "
        "models or layers, and JAX models must be Flax modules"
    )


__all__ = [
    "Backend",
    "TapRecorder",
    "framework_roots",
    "installed",
    "known",
    "module_roots",
    "resolve",
]
