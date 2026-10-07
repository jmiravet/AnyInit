"""The contract a framework adapter has to satisfy.

A backend traces a model into the shared IR, moves weights in and out in canonical layout,
evaluates a native callable on an array, and runs a forward pass reporting statistics
rather than tensors.  Everything numerical lives in :mod:`anyinit.core`.

Reporting statistics rather than tensors is what keeps the boundary intact: the core never
holds a framework object and so never reaches for a framework operation.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.graph import ModelGraph
from ..core.moments import MomentState
from ..core.registry import REGISTRY


def module_roots(obj: Any) -> set[str]:
    """Top-level package names appearing anywhere in an object's class hierarchy.

    Accepts an instance, a class, or a function.  Backend detection reads this rather than
    using ``isinstance``, which would require importing every supported framework to ask
    the question.  A model's own framework is necessarily imported already.
    """
    hierarchy = obj.__mro__ if isinstance(obj, type) else type(obj).__mro__
    roots = set()
    for cls in hierarchy:
        module = getattr(cls, "__module__", "") or ""
        roots.add(module.split(".")[0])
    if not isinstance(obj, type):
        # A plain function carries no MRO; its defining module is the useful signal.
        own = getattr(obj, "__module__", "") or ""
        if own:
            roots.add(own.split(".")[0])
    return roots


def framework_roots(obj: Any) -> set[str]:
    """Top-level packages a registered activation is written against.

    A class or instance answers through its hierarchy.  A plain function has no telling
    hierarchy, so the globals its code refers to are inspected instead: ``torch.relu(x)``
    names ``torch``, ``jax.nn.relu(x)`` names ``jax``.
    """
    roots = set(module_roots(obj))
    code = getattr(obj, "__code__", None)
    namespace = getattr(obj, "__globals__", None)
    if code is not None and isinstance(namespace, dict):
        for name in code.co_names:
            target = namespace.get(name)
            if target is None:
                continue
            module = getattr(target, "__name__", None) if isinstance(target, type(sys)) else None
            module = module or getattr(target, "__module__", None) or ""
            if module:
                roots.add(module.split(".")[0])
    return roots


class Backend(ABC):
    """Adapter between one framework and AnyInit's core."""

    #: Short identifier used in reports and profile cache keys.
    name: str = "base"
    #: Package names this backend covers, for error messages.
    frameworks: tuple[str, ...] = ()
    #: The package the adapter imports.  Its presence is what makes the backend usable.
    requires: str = ""
    #: What to ``pip install`` to make this backend usable.
    install: str = ""

    @property
    def label(self) -> str:
        """How the report names this backend; may add what runs underneath."""
        return self.name

    # ------------------------------------------------------------- discovery

    @classmethod
    def installed(cls) -> bool:
        """Whether the framework this backend adapts can be imported here.

        Locates the package without executing it, so asking never imports a framework.
        """
        if not cls.requires:
            return False
        # A framework installed after this process started must still be found.
        importlib.invalidate_caches()
        try:
            return importlib.util.find_spec(cls.requires) is not None
        except (ImportError, ValueError):
            return False

    @classmethod
    def owns(cls, obj: Any) -> bool:
        """Whether a registered activation is written against this backend's framework."""
        return bool(framework_roots(obj) & set(cls.frameworks))

    @classmethod
    def registered_types(cls) -> dict[Any, str]:
        """Registered activation classes that belong to this framework."""
        return {t: n for t, n in REGISTRY.native_types().items() if cls.owns(t)}

    @classmethod
    def registered_callables(cls) -> dict[Any, str]:
        """Registered activation functions that belong to this framework."""
        return {f: n for f, n in REGISTRY.native_callables().items() if cls.owns(f)}

    @staticmethod
    @abstractmethod
    def handles(model: Any) -> bool:
        """Whether this backend recognizes ``model``, without importing anything."""

    # ----------------------------------------------------------------- graph

    @abstractmethod
    def build_graph(self, model: Any, input_spec: Any = None) -> ModelGraph:
        """Trace ``model`` into the shared IR."""

    # ------------------------------------------------------- numerics bridge

    @abstractmethod
    def eval_elementwise(self, fn: Any, x: np.ndarray) -> np.ndarray:
        """Apply a native callable to quadrature abscissas and return an array.

        Only needed for activations the user registered as native code; builtins are
        profiled from their NumPy references and never reach a backend.
        """

    # ------------------------------------------------------------ parameters

    @abstractmethod
    def read_weight(self, handle: Any) -> np.ndarray:
        """Return the current weight, in canonical ``(fan_out, fan_in, *receptive)`` layout."""

    @abstractmethod
    def write_weight(self, handle: Any, weight: np.ndarray) -> None:
        """Store a canonical-layout weight, converting to the framework's own order."""

    @abstractmethod
    def write_bias(self, handle: Any, bias: np.ndarray) -> None:
        """Store a bias vector."""

    def write_gain(self, handle: Any, gain: float) -> None:
        """Set a normalization layer's scale parameter.

        Assignment rather than multiplication, so repeated calls are idempotent.
        """
        raise NotImplementedError(f"{self.name} backend cannot set normalization gains")

    # -------------------------------------------------------------- measuring

    @abstractmethod
    def forward_taps(
        self, model: Any, inputs: Any, taps: Sequence[str], *, training: bool = True
    ) -> dict[str, MomentState]:
        """Run one forward pass and report moments at the named nodes.

        ``training`` must normally be true: at initialization a batch-norm layer's running
        statistics are still (0, 1), so in inference mode it scales by gamma instead of
        normalizing.  Any running statistics the pass disturbs are restored.
        """

    def make_inputs(self, input_spec: Any, seed: int | None = None) -> Any:
        """Turn ``input_spec`` into something the model can be called with."""
        raise NotImplementedError(f"{self.name} backend cannot synthesize inputs")

    def input_moments(self, inputs: Any) -> MomentState | None:
        """Raw moments of a batch, or ``None`` when they cannot be read.

        The boundary condition of the analytic recursion.  Goes through NumPy, so a backend
        overrides this only when its tensors need unwrapping first.
        """
        try:
            array = np.asarray(inputs, dtype=np.float64).ravel()
        except Exception:
            return None
        if array.size == 0 or not np.all(np.isfinite(array)):
            return None
        return MomentState(float(array.mean()), float((array**2).mean()), float((array**4).mean()))

    # --------------------------------------------------------------- lifecycle

    def begin(self, model: Any, params: Any = None) -> None:  # noqa: B027
        """Prepare for a run, before any weights are written.

        ``params`` carries the parameter tree for functional frameworks.
        """

    def finalize(self, model: Any) -> Any:
        """Finish a run, returning new parameters for a functional backend."""
        return None

    def unscaled_weights(self, model: Any, graph: ModelGraph) -> list[tuple[str, str]]:
        """Weight tensors of two or more dimensions the graph does not cover.

        Returned as ``(name, reason)`` pairs so the report can say exactly what was left
        as the framework initialized it, and why.
        """
        return []

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"


class TapRecorder:
    """Accumulates raw moments across batches, which average linearly in the count."""

    def __init__(self) -> None:
        self._sums: dict[str, tuple[float, float, float, int]] = {}

    def add(self, node_id: str, values: np.ndarray) -> None:
        """Accumulate one batch of values for a node."""
        flat = np.asarray(values, dtype=np.float64).ravel()
        if flat.size == 0:
            return
        squares = flat * flat
        self.add_sums(
            node_id,
            float(np.sum(flat)),
            float(np.sum(squares)),
            float(np.sum(squares * squares)),
            int(flat.size),
        )

    def add_sums(self, node_id: str, s1: float, s2: float, s4: float, n: int) -> None:
        """Accumulate power sums computed elsewhere, such as on the framework's device."""
        if node_id in self._sums:
            a, b, c, m = self._sums[node_id]
            self._sums[node_id] = (a + s1, b + s2, c + s4, m + n)
        else:
            self._sums[node_id] = (s1, s2, s4, n)

    def result(self) -> dict[str, MomentState]:
        """Pooled moments per node."""
        out: dict[str, MomentState] = {}
        for node_id, (s1, s2, s4, n) in self._sums.items():
            if n == 0:
                continue
            out[node_id] = MomentState(s1 / n, s2 / n, s4 / n)
        return out

    def counts(self) -> dict[str, int]:
        """Sample count behind each measurement."""
        return {node_id: n for node_id, (_, _, _, n) in self._sums.items()}
