"""JAX adapter, by way of Flax.

JAX parameters are immutable, so this backend works on its own copy and hands back a new
tree as ``report.params``; :func:`anyinit.initialize_params` wraps that into the shape JAX
code usually takes.

The call order is recovered by instrumentation rather than from a jaxpr: inside a context
manager the Flax layer classes and ``jax.nn`` activations are wrapped, the model is applied
to a probe of at most 16 input rows, and each layer reports ``scope.path``, which is its
key in the parameter tree.  A jaxpr is too far from the source to read activations off --
most lower to opaque wrappers, and a bias add is indistinguishable from a residual one.

An activation stored as a module field, captured in a closure or written as a lambda never
passes through ``jax.nn`` and so is not seen being called.  The probe's values identify it
instead: wherever one layer's output reaches the next layer transformed, the transform is
matched against the known and registered activations.

This yields an ordered chain rather than a DAG, since nothing intercepts ``+``.  The graph
is marked ``linear`` and the report says so; branching architectures are scaled as if they
were sequential.
"""

from __future__ import annotations

import contextlib
import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core import activations as builtin_activations
from ..core import fan as fanmod
from ..core.graph import FIDELITY_LINEAR, ModelGraph, Node, NodeKind
from ..core.moments import MomentState
from ..core.registry import REGISTRY, ActivationRef
from ..errors import TraceError
from .base import Backend, TapRecorder, module_roots

#: Flax layer class -> (ParamSpec kind, parameter name).
_LAYERS: dict[str, tuple[str, str]] = {
    "Dense": (fanmod.DENSE, "kernel"),
    "DenseGeneral": (fanmod.DENSE, "kernel"),
    "Conv": (fanmod.CONV, "kernel"),
    "ConvLocal": (fanmod.CONV, "kernel"),
    "ConvTranspose": (fanmod.CONV_TRANSPOSE, "kernel"),
    "Embed": (fanmod.EMBEDDING, "embedding"),
}

_NORMS = frozenset({"LayerNorm", "BatchNorm", "GroupNorm", "RMSNorm", "InstanceNorm"})

#: Activation functions to intercept.  Both ``jax.nn`` and the ``flax.linen`` re-exports
#: are patched, so either import style is seen.
_ACTIVATION_ATTRS: tuple[str, ...] = (
    "relu",
    "relu6",
    "leaky_relu",
    "elu",
    "selu",
    "celu",
    "gelu",
    "sigmoid",
    "silu",
    "swish",
    "mish",
    "softplus",
    "soft_sign",
    "tanh",
    "hard_tanh",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "softmax",
    "log_softmax",
    "identity",
)

#: JAX names that differ from AnyInit's canonical ones.
_CANONICAL: dict[str, str] = {
    "swish": "silu",
    "soft_sign": "softsign",
    "hard_tanh": "hardtanh",
    "hard_sigmoid": "hardsigmoid",
    "hard_silu": "hardswish",
    "hard_swish": "hardswish",
}


@dataclass(frozen=True)
class FlaxHandle:
    """Which entry of the flat parameter dict a scalable node owns."""

    path: tuple[str, ...]
    kind: str

    @property
    def bias_path(self) -> tuple[str, ...]:
        """Path of the bias sitting alongside this weight."""
        return (*self.path[:-1], "bias")


class FlaxBackend(Backend):
    """AnyInit adapter for Flax modules."""

    name = "jax"
    frameworks = ("flax", "jax")
    requires = "flax"
    install = "jax flax"

    def __init__(self) -> None:
        import jax  # Imported here, never at module scope.
        import jax.numpy as jnp

        self._jax = jax
        self._jnp = jnp
        self._flat: dict[tuple[str, ...], np.ndarray] = {}
        self._collections: dict[str, Any] = {}
        self._trace: list[tuple[int, str, str, tuple[str, ...]]] = []
        self._inferred: dict[int, str] = {}
        self.last_counts: dict[str, int] = {}

    @staticmethod
    def handles(model: Any) -> bool:
        """Recognize a Flax module from its class hierarchy."""
        roots = module_roots(model)
        return "flax" in roots and hasattr(model, "apply")

    def begin(self, model: Any, params: Any = None) -> None:
        """Take a working copy of the incoming parameter tree."""
        if params is not None:
            self._absorb(params)

    def _absorb(self, params: Any) -> None:
        """Flatten an incoming parameter tree into a mutable NumPy working copy."""
        from flax.traverse_util import flatten_dict

        mapping = dict(params)
        trainable = mapping.get("params", mapping)
        self._collections = {k: v for k, v in mapping.items() if k != "params"}
        self._has_params_key = "params" in mapping
        self._flat = {
            tuple(str(p) for p in path): np.asarray(value, dtype=np.float64)
            for path, value in flatten_dict(dict(trainable)).items()
        }

    # ----------------------------------------------------------------- graph

    def build_graph(self, model: Any, input_spec: Any = None) -> ModelGraph:
        """Trace the module by instrumentation, discovering parameters if needed."""
        if not self._flat:
            if input_spec is None:
                raise TraceError(
                    "the JAX backend needs either an existing parameter tree (params=...) "
                    "or input_spec, so it can call model.init to discover the parameters"
                )
            self._absorb(model.init(self._jax.random.key(0), self._dummy(input_spec)))

        dummy = self._dummy(input_spec) if input_spec is not None else None
        if dummy is None:
            raise TraceError(
                "the JAX backend needs input_spec to trace the model; pass a shape or a batch"
            )

        probe = self._probe(dummy)
        inputs: dict[int, np.ndarray] = {}
        outputs: dict[int, np.ndarray] = {}
        with _instrumented(
            observe=lambda i, v: outputs.__setitem__(i, np.asarray(v)),
            observe_input=lambda i, v: inputs.__setitem__(i, np.asarray(v)),
        ) as trace:
            final = _first(self._apply(model, probe))
        self._trace = list(trace)
        if not self._trace:
            raise TraceError(
                "nothing was recorded while tracing this Flax module. AnyInit intercepts "
                "flax.linen layers and jax.nn activations; a model built from raw jnp "
                "operations is invisible to it"
            )
        if final is not None and self._trace:
            inputs[_MODEL_OUTPUT] = np.asarray(final)
        self._inferred, notes = self._infer_activations(inputs, outputs)
        return self._graph_from_trace(notes)

    def _probe(self, batch: Any) -> Any:
        """A few rows of the input, enough to read activations off their values."""
        array = self._jnp.asarray(batch)
        return array[:16] if array.ndim > 0 and array.shape[0] > 16 else array

    def _apply(self, model: Any, batch: Any, training: bool = True) -> Any:
        mutable = ["batch_stats"] if training and "batch_stats" in self._collections else False
        return model.apply(self._pytree(), batch, **({"mutable": mutable} if mutable else {}))

    def _infer_activations(
        self, inputs: dict[int, np.ndarray], outputs: dict[int, np.ndarray]
    ) -> tuple[dict[int, str], list[str]]:
        """Identify activations applied between consecutive layers, by their values.

        Instrumenting ``jax.nn`` only sees calls made through that namespace.  The usual
        Flax idioms -- an activation stored as a module field, captured in a closure, or
        written as a lambda -- call the function object directly and are never seen.  So
        every pair of consecutive layers with nothing recorded between them is checked:
        if the second layer's input is a known activation applied elementwise to the first
        layer's output, that activation is inserted.  Returns the activations found, keyed
        by the trace index of the layer they feed, and notes for anything unidentifiable.
        """
        found: dict[int, str] = {}
        notes: list[str] = []
        layers = [entry for entry in self._trace if entry[1] in ("layer", "norm")]
        if _MODEL_OUTPUT in inputs:
            # The model's own output stands in for a layer after the last one, so an
            # activation applied last is caught too.
            layers.append((_MODEL_OUTPUT, "output", "output", ()))
        recorded = {entry[0] for entry in self._trace if entry[1] == "activation"}
        for previous, current in itertools.pairwise(layers):
            upper = current[0] if current[0] != _MODEL_OUTPUT else float("inf")
            if any(previous[0] < index < upper for index in recorded):
                continue
            z, x = outputs.get(previous[0]), inputs.get(current[0])
            if z is None or x is None:
                continue
            if z.shape != x.shape:
                if z.size != x.size:
                    continue  # pooling, slicing or a merge: not an elementwise step
                x = x.reshape(z.shape)
            if _close(z, x):
                continue  # genuinely linear between the two layers
            name = self._identify(z, x)
            if name is not None:
                found[current[0]] = name
            else:
                notes.append(
                    f"between {'/'.join(previous[3]) or previous[2]} and "
                    f"{'/'.join(current[3]) or current[2]} the signal is transformed by "
                    "something AnyInit could not identify; treated as linear"
                )
        return found, notes

    def _try_native(self, fn: Any, z: np.ndarray) -> np.ndarray:
        """Apply a registered JAX function, or return NaNs if it does not accept ``z``."""
        try:
            return np.asarray(fn(self._jnp.asarray(z)))
        except Exception:
            return np.full(z.shape, np.nan)

    def _identify(self, z: np.ndarray, x: np.ndarray) -> str | None:
        """Name of the activation mapping ``z`` to ``x`` elementwise, if there is one."""
        z64 = z.astype(np.float64)
        for name in _IDENTIFIABLE:
            candidate = builtin_activations.BUILTIN[name](z64)
            if _close(candidate, x):
                return name
        for fn, name in FlaxBackend.registered_callables().items():
            if _close(self._try_native(fn, z), x):
                return name
        for name in REGISTRY.custom_names:
            spec = REGISTRY.spec(name)
            if spec is not None and spec.numpy_fn is not None and _close(spec.numpy_fn(z64), x):
                return name
        return None

    def _graph_from_trace(self, notes: Sequence[str] = ()) -> ModelGraph:
        nodes: list[Node] = [Node("input", NodeKind.INPUT, "input")]
        previous = "input"

        for index, kind, op, path in self._trace:
            inferred = self._inferred.get(index)
            if inferred is not None:
                nodes.append(
                    Node(
                        _inferred_id(inferred, index),
                        NodeKind.ACTIVATION,
                        inferred,
                        (previous,),
                        meta={"activation": ActivationRef(inferred), "input_of": index},
                    )
                )
                previous = nodes[-1].id
            nid = "/".join(path) if path else f"{op}_{index}"
            if kind == "activation":
                canonical = _CANONICAL.get(op, op)
                nodes.append(
                    Node(
                        nid,
                        NodeKind.ACTIVATION,
                        canonical,
                        (previous,),
                        meta={"activation": ActivationRef(canonical), "trace_index": index},
                    )
                )
            elif kind == "norm":
                scale_path = (*path, "scale")
                if scale_path in self._flat:
                    nodes.append(
                        Node(
                            nid,
                            NodeKind.NORMALIZATION,
                            op,
                            (previous,),
                            spec=fanmod.ParamSpec(fanmod.NORM, self._flat[scale_path].shape),
                            handle=FlaxHandle(scale_path, op),
                            meta={"path": nid, "trace_index": index, "affine": True},
                        )
                    )
                else:
                    nodes.append(
                        Node(
                            nid,
                            NodeKind.NORMALIZATION,
                            op,
                            (previous,),
                            meta={"path": nid, "trace_index": index, "affine": False},
                        )
                    )
            else:
                spec_kind, param_name = _LAYERS[op]
                weight_path = (*path, param_name)
                native = self._flat.get(weight_path)
                if native is None:
                    nodes.append(
                        Node(
                            nid,
                            NodeKind.OTHER,
                            op,
                            (previous,),
                            meta={"path": nid, "trace_index": index},
                        )
                    )
                else:
                    nodes.append(
                        Node(
                            nid,
                            NodeKind.PARAMETRIC,
                            op,
                            (previous,),
                            spec=_spec_for(
                                spec_kind, native.shape, has_bias=(*path, "bias") in self._flat
                            ),
                            handle=FlaxHandle(weight_path, spec_kind),
                            meta={"path": nid, "trace_index": index},
                        )
                    )
            previous = nodes[-1].id

        trailing = self._inferred.get(_MODEL_OUTPUT)
        if trailing is not None:
            nodes.append(
                Node(
                    _inferred_id(trailing, _MODEL_OUTPUT),
                    NodeKind.ACTIVATION,
                    trailing,
                    (previous,),
                    meta={"activation": ActivationRef(trailing), "input_of": _MODEL_OUTPUT},
                )
            )
            previous = nodes[-1].id
        nodes.append(Node("output", NodeKind.OUTPUT, "output", (previous,)))
        return ModelGraph(
            nodes,
            fidelity=FIDELITY_LINEAR,
            notes=(
                "the JAX backend recovers call order, not graph structure, so residual "
                "additions and other merges are invisible and each layer is scaled as if "
                "the model were sequential",
                *notes,
            ),
        )

    def unscaled_weights(self, model: Any, graph: ModelGraph) -> list[tuple[str, str]]:
        """Every ``>= 2``-d parameter no graph node writes, with the reason."""
        covered = {
            node.handle.path for node in graph.scalable if isinstance(node.handle, FlaxHandle)
        }
        return [
            ("/".join(path), "not reached by a recognized Flax layer during tracing")
            for path, value in self._flat.items()
            if value.ndim >= 2 and path not in covered
        ]

    # ------------------------------------------------------- numerics bridge

    def eval_elementwise(self, fn: Any, x: np.ndarray) -> np.ndarray:
        """Apply a JAX callable to quadrature abscissas."""
        out = fn(self._jnp.asarray(x, dtype=self._jnp.float64))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return np.asarray(out, dtype=np.float64)

    # ------------------------------------------------------------ parameters

    def read_weight(self, handle: Any) -> np.ndarray:
        """Read a parameter, converting Flax's axis order to canonical."""
        return _to_canonical(handle.kind, self._flat[handle.path])

    def write_weight(self, handle: Any, weight: np.ndarray) -> None:
        """Write a canonical weight into the working copy, in Flax's axis order."""
        native_shape = self._flat[handle.path].shape
        self._flat[handle.path] = _from_canonical(handle.kind, weight, native_shape)

    def write_bias(self, handle: Any, bias: np.ndarray) -> None:
        """Write a bias into the working copy."""
        path = handle.bias_path
        existing = self._flat.get(path)
        if existing is None:
            return
        flat = np.asarray(bias, dtype=np.float64).ravel()[: existing.size]
        self._flat[path] = flat.reshape(existing.shape)

    def write_gain(self, handle: Any, gain: float) -> None:
        """Set a normalization scale, and zero its offset."""
        existing = self._flat.get(handle.path)
        if existing is None:
            return
        # Assignment, not multiplication, so repeated calls are idempotent.
        self._flat[handle.path] = np.full(existing.shape, float(gain))
        bias_path = handle.bias_path
        if bias_path in self._flat:
            self._flat[bias_path] = np.zeros(self._flat[bias_path].shape)

    # -------------------------------------------------------------- measuring

    def forward_taps(
        self, model: Any, inputs: Any, taps: Sequence[str], *, training: bool = True
    ) -> dict[str, MomentState]:
        """Run the model with the instrumentation recording output moments.

        The same interception that recovers the call order reads the values flowing through
        it, which is how activations -- plain function calls, with nothing to hook -- get
        tapped.
        """
        wanted = set(taps)
        recorder = TapRecorder()
        index_to_id = dict(self._trace_ids())
        input_to_id = {index: _inferred_id(name, index) for index, name in self._inferred.items()}

        def record(table: dict[int, str], index: int, value: Any) -> None:
            nid = table.get(index)
            if nid is None or nid not in wanted:
                return
            array = np.asarray(value, dtype=np.float64)
            if array.size:
                recorder.add(nid, array)

        with _instrumented(
            observe=lambda i, v: record(index_to_id, i, v),
            observe_input=lambda i, v: record(input_to_id, i, v),
        ):
            final = _first(self._apply(model, self._jnp.asarray(inputs), training))
            self._jax.block_until_ready(final)
        record(input_to_id, _MODEL_OUTPUT, final)
        self.last_counts = recorder.counts()
        return recorder.result()

    def _trace_ids(self) -> Iterator[tuple[int, str]]:
        for index, _kind, op, path in self._trace:
            yield index, ("/".join(path) if path else f"{op}_{index}")

    # --------------------------------------------------------------- lifecycle

    def finalize(self, model: Any) -> Any:
        """Rebuild the parameter tree, in the shape it arrived in."""
        from flax.traverse_util import unflatten_dict

        params = unflatten_dict(
            {
                path: self._jnp.asarray(value, dtype=self._jnp.float32)
                for path, value in self._flat.items()
            }
        )
        if getattr(self, "_has_params_key", True):
            return {"params": params, **self._collections}
        return params

    def _pytree(self) -> Any:
        return self.finalize(None)

    def make_inputs(self, input_spec: Any, seed: int | None = None) -> Any:
        """Build a JAX batch from a shape, a callable or an array."""
        return self._dummy(input_spec, seed)

    def _dummy(self, input_spec: Any, seed: int | None = None) -> Any:
        if callable(input_spec) and not isinstance(input_spec, (tuple, list)):
            return input_spec()
        if isinstance(input_spec, (tuple, list)) and all(isinstance(d, int) for d in input_spec):
            rng = np.random.default_rng(0 if seed is None else int(seed))
            return self._jnp.asarray(
                rng.standard_normal(tuple(input_spec)), dtype=self._jnp.float32
            )
        return self._jnp.asarray(input_spec)


# ------------------------------------------------------- instrumentation


@contextlib.contextmanager
def _instrumented(
    observe: Any = None, observe_input: Any = None
) -> Iterator[list[tuple[int, str, str, tuple[str, ...]]]]:
    """Temporarily wrap Flax layers and JAX activations to record the call sequence.

    Everything patched is restored on exit, including when the body raises.
    """
    import flax.linen as nn
    import jax.nn as jnn
    import jax.numpy as jnp

    trace: list[tuple[int, str, str, tuple[str, ...]]] = []
    counter = [0]
    saved: list[tuple[Any, str, Any]] = []

    def record(kind: str, op: str, path: tuple[str, ...]) -> int:
        index = counter[0]
        counter[0] += 1
        trace.append((index, kind, op, path))
        return index

    def wrap_layer(cls: Any, op: str, kind: str) -> None:
        original = cls.__call__
        saved.append((cls, "__call__", original))

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            scope = getattr(self, "scope", None)
            path = tuple(str(p) for p in scope.path) if scope is not None else ()
            index = record(kind, op, path)
            if observe_input is not None and args:
                observe_input(index, args[0])
            out = original(self, *args, **kwargs)
            if observe is not None:
                observe(index, out)
            return out

        cls.__call__ = wrapper

    def wrap_function(holder: Any, attr: str) -> None:
        original = getattr(holder, attr, None)
        if original is None or not callable(original):
            return
        saved.append((holder, attr, original))

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            index = record("activation", attr, ())
            out = original(*args, **kwargs)
            if observe is not None:
                observe(index, out)
            return out

        setattr(holder, attr, wrapper)

    try:
        for op in _LAYERS:
            cls = getattr(nn, op, None)
            if cls is not None:
                wrap_layer(cls, op, "layer")
        for op in _NORMS:
            cls = getattr(nn, op, None)
            if cls is not None:
                wrap_layer(cls, op, "norm")
        for name, canonical in FlaxBackend.registered_types().items():
            if isinstance(name, type):
                wrap_layer(name, canonical, "activation")
        for attr in _ACTIVATION_ATTRS:
            wrap_function(jnn, attr)
            wrap_function(nn, attr)
        wrap_function(jnp, "tanh")
        for fn in FlaxBackend.registered_callables():
            holder = _holder_of(fn)
            if holder is not None:
                wrap_function(holder, fn.__name__)
        yield trace
    finally:
        for holder, attr, original in reversed(saved):
            setattr(holder, attr, original)


#: Builtins tried, in order, when identifying an activation from its values.  Aliases are
#: left out, and where two functions agree on typical inputs the commoner comes first.
_IDENTIFIABLE: tuple[str, ...] = (
    "relu",
    "gelu_tanh",
    "gelu",
    "silu",
    "tanh",
    "sigmoid",
    "elu",
    "selu",
    "celu",
    "softplus",
    "mish",
    "softsign",
    "hardtanh",
    "hardsigmoid",
    "hardswish",
    "relu6",
    "leaky_relu",
)


#: Trace key standing for the model's output, so a final activation can be identified.
_MODEL_OUTPUT = -1


def _inferred_id(name: str, index: int) -> str:
    """Node id of an activation identified from values, by the layer it feeds."""
    return f"{name}_before_{'output' if index == _MODEL_OUTPUT else index}"


def _first(result: Any) -> Any:
    """The model output, unwrapping the ``(output, state)`` pair a mutable apply returns."""
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[0]
    return result


def _close(a: np.ndarray, b: np.ndarray) -> bool:
    """Equality to float32 working precision, scaled by the magnitude of the values."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    scale = max(float(np.abs(b).max(initial=0.0)), 1.0)
    return bool(np.allclose(a, b, rtol=1e-5, atol=1e-6 * scale))


def _holder_of(fn: Any) -> Any:
    """Return the module a registered function lives in, so it can be patched in place."""
    import sys

    module_name = getattr(fn, "__module__", None)
    return sys.modules.get(module_name) if module_name else None


# ------------------------------------------------------------------ layouts


def _spec_for(kind: str, native_shape: tuple[int, ...], *, has_bias: bool) -> fanmod.ParamSpec:
    if kind == fanmod.EMBEDDING:
        return fanmod.ParamSpec(kind, tuple(int(s) for s in native_shape))
    canonical = _to_canonical_shape(kind, tuple(int(s) for s in native_shape))
    return fanmod.ParamSpec(kind, canonical, has_bias=has_bias)


def _to_canonical_shape(kind: str, native: tuple[int, ...]) -> tuple[int, ...]:
    if kind == fanmod.EMBEDDING:
        return native
    if len(native) == 2:  # Dense: (in, out)
        return (native[1], native[0])
    *kernel, in_ch, out_ch = native  # Conv / ConvTranspose: (*kernel, in, out)
    return (out_ch, in_ch, *kernel)


def _to_canonical(kind: str, native: np.ndarray) -> np.ndarray:
    if kind == fanmod.EMBEDDING or native.ndim < 2:
        return native
    if native.ndim == 2:
        return native.T
    return np.moveaxis(native, (-1, -2), (0, 1))


def _from_canonical(kind: str, weight: np.ndarray, native_shape: tuple[int, ...]) -> np.ndarray:
    if kind == fanmod.EMBEDDING or weight.ndim < 2:
        return weight.reshape(native_shape)
    if weight.ndim == 2:
        return weight.T.reshape(native_shape)
    return np.moveaxis(weight, (0, 1), (-1, -2)).reshape(native_shape)
