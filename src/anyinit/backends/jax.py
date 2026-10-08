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
from .base import Backend, TapRecorder, input_rng, module_roots

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
        self._superseded: set[int] = set()
        self._scales: dict[int, float] = {}
        self._output_leaf: int | None = None
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
        with (
            self._probe_weights(),
            _instrumented(
                observe=lambda i, v: outputs.__setitem__(i, np.asarray(v)),
                observe_input=lambda i, v: inputs.__setitem__(i, np.asarray(v)),
            ) as trace,
        ):
            final = _first(self._apply(model, probe))
        self._trace = list(trace)
        if not self._trace:
            raise TraceError(
                "nothing was recorded while tracing this Flax module. AnyInit intercepts "
                "flax.linen layers and jax.nn activations; a model built from raw jnp "
                "operations is invisible to it"
            )
        leaves = [np.asarray(leaf) for leaf in self._jax.tree.leaves(final)]
        self._inferred, notes = self._infer_activations(inputs, outputs, leaves)
        return self._graph_from_trace(notes)

    @contextlib.contextmanager
    def _probe_weights(self) -> Iterator[None]:
        """Trace with unit-scale weights in place of the ones the model came with.

        Activations are read off the probe's values, and the incoming weights may shrink
        the signal below anything a comparison can resolve -- ``normal(0.02)`` leaves
        about 1e-7 after ten layers -- or saturate it.  Weights drawn with variance
        ``1/fan_in`` and zero biases are close to what AnyInit is about to write.
        """
        original = self._flat
        rng = np.random.default_rng(0)
        provisional = dict(original)
        for path, value in original.items():
            if path[-1] in _WEIGHT_NAMES and value.ndim >= 2:
                fan = 1 if path[-1] == "embedding" else int(np.prod(value.shape[:-1]))
                provisional[path] = rng.standard_normal(value.shape) / np.sqrt(max(fan, 1))
            elif path[-1] == "bias" and (*path[:-1], "kernel") in original:
                provisional[path] = np.zeros(value.shape)
        self._flat = provisional
        try:
            yield
        finally:
            self._flat = original

    def _probe(self, batch: Any) -> Any:
        """A few rows of the input, enough to read activations off their values."""
        array = self._jnp.asarray(batch)
        return array[:16] if array.ndim > 0 and array.shape[0] > 16 else array

    def _apply(self, model: Any, batch: Any, training: bool = True) -> Any:
        mutable = ["batch_stats"] if training and "batch_stats" in self._collections else False
        return model.apply(self._pytree(), batch, **({"mutable": mutable} if mutable else {}))

    def _infer_activations(
        self,
        inputs: dict[int, np.ndarray],
        outputs: dict[int, np.ndarray],
        final: Sequence[np.ndarray] = (),
    ) -> tuple[dict[int, str], list[str]]:
        """Identify activations applied between consecutive layers, by their values.

        Instrumenting ``jax.nn`` only sees calls made through that namespace.  The usual
        Flax idioms -- an activation stored as a module field, captured in a closure, or
        written as a lambda -- call the function object directly and are never seen.  So
        every pair of consecutive layers is checked: if the second layer's input is not
        what the first layer's output became through the activations recorded between
        them, the transform is matched against the known and registered activations, and
        what it identifies takes the place of what was recorded.  A registered function
        that calls ``jnp.tanh`` inside is caught this way, rather than taken for ``tanh``.
        A transform that matches no activation is tried as a constant factor, such as the
        ``sqrt(d_model)`` a language model multiplies its lookup by, kept in ``_scales``.

        ``final`` holds the leaves of the model's output, which stands in for a layer after
        the last one, so an activation applied last is caught too.  Returns the
        activations found, keyed by the trace index of the layer they feed, and notes for
        anything unidentifiable.
        """
        found: dict[int, str] = {}
        notes: list[str] = []
        self._superseded = set()
        self._scales = {}
        self._output_leaf = None
        layers = [entry for entry in self._trace if entry[1] in ("layer", "norm")]
        if final:
            layers.append((_MODEL_OUTPUT, "output", "output", ()))
        recorded = [entry[0] for entry in self._trace if entry[1] == "activation"]
        for previous, current in itertools.pairwise(layers):
            upper = current[0] if current[0] != _MODEL_OUTPUT else float("inf")
            between = [index for index in recorded if previous[0] < index < upper]
            z = outputs.get(previous[0])
            if z is None:
                continue
            if current[0] == _MODEL_OUTPUT:
                # A model returning several arrays is matched leaf by leaf, latest first.
                candidates = list(enumerate(final))[::-1]
            else:
                candidates = [(0, inputs[current[0]])] if current[0] in inputs else []
            last = outputs.get(between[-1]) if between else None
            seen = False
            for leaf, x in candidates:
                if z.size != x.size:
                    continue  # pooling, slicing or a merge: not an elementwise step
                seen = True
                x = x.reshape(z.shape)
                if last is not None and (last.size != x.size or _close(last.reshape(x.shape), x)):
                    verdict: str | None = ""  # the recorded activations account for it
                elif last is None and _close(z, x):
                    verdict = ""  # genuinely linear between the two layers
                else:
                    verdict = self._identify(z, x)
                    factor = None if verdict else _scalar_ratio(z if last is None else last, x)
                    if factor is not None:
                        self._scales[current[0]] = factor
                        verdict = ""
                if verdict is None:
                    continue
                if current[0] == _MODEL_OUTPUT:
                    self._output_leaf = leaf
                if verdict:
                    found[current[0]] = verdict
                    self._superseded.update(between)
                break
            else:
                if seen:
                    where = (
                        f"between {'/'.join(previous[3]) or previous[2]} and "
                        f"{'/'.join(current[3]) or current[2]}"
                    )
                    notes.append(
                        f"{where} the signal is transformed by something AnyInit could not "
                        + (
                            "identify, beyond the activations it saw called; scaled for those"
                            if between
                            else "identify; treated as linear"
                        )
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
            if spec is None:
                continue
            if spec.numpy_fn is not None:
                candidate = spec.numpy_fn(z64)
            elif spec.native_type is None and callable(spec.native_fn):
                candidate = self._try_native(spec.native_fn, z)
            else:
                continue
            if _close(candidate, x):
                return name
        return None

    def _graph_from_trace(self, notes: Sequence[str] = ()) -> ModelGraph:
        nodes: list[Node] = [Node("input", NodeKind.INPUT, "input")]
        previous = "input"

        for index, kind, op, path in self._trace:
            if index in self._superseded:
                continue
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
            if index in self._scales:
                nodes.append(_scale_node(index, self._scales[index], previous))
                previous = nodes[-1].id
            nid = _trace_id(index, op, path)
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
            elif op == _ATTEND:
                nodes.append(_tied_readout(nid, path, previous, index, self._flat))
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
        if _MODEL_OUTPUT in self._scales:
            nodes.append(_scale_node(_MODEL_OUTPUT, self._scales[_MODEL_OUTPUT], previous))
            previous = nodes[-1].id
        nodes.append(Node("output", NodeKind.OUTPUT, "output", (previous,)))
        return ModelGraph(
            _mark_tied(nodes),
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
        input_to_id.update({index: _inferred_id("scale", index) for index in self._scales})

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
        if self._output_leaf is not None:
            record(input_to_id, _MODEL_OUTPUT, self._jax.tree.leaves(final)[self._output_leaf])
        self.last_counts = recorder.counts()
        return recorder.result()

    def _trace_ids(self) -> Iterator[tuple[int, str]]:
        for index, _kind, op, path in self._trace:
            yield index, _trace_id(index, op, path)

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
            rng = input_rng(seed)
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
    # Activations under way.  One activation calling another -- a registered function
    # built on jnp.tanh, or jax.nn.gelu on tanh -- is a single step, recorded once.
    depth = [0]
    saved: list[tuple[Any, str, Any]] = []

    def record(kind: str, op: str, path: tuple[str, ...]) -> int:
        index = counter[0]
        counter[0] += 1
        trace.append((index, kind, op, path))
        return index

    def wrap_layer(cls: Any, op: str, kind: str, method: str = "__call__") -> None:
        original = getattr(cls, method)
        saved.append((cls, method, original))

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if depth[0]:
                return original(self, *args, **kwargs)
            scope = getattr(self, "scope", None)
            path = tuple(str(p) for p in scope.path) if scope is not None else ()
            index = record(kind, op, path)
            if observe_input is not None and args:
                observe_input(index, args[0])
            if kind == "activation":
                depth[0] += 1
            try:
                out = original(self, *args, **kwargs)
            finally:
                if kind == "activation":
                    depth[0] -= 1
            if observe is not None:
                observe(index, out)
            return out

        setattr(cls, method, wrapper)

    def wrap_function(holder: Any, attr: str) -> None:
        original = getattr(holder, attr, None)
        if original is None or not callable(original):
            return
        saved.append((holder, attr, original))

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if depth[0]:
                return original(*args, **kwargs)
            index = record("activation", attr, ())
            depth[0] += 1
            try:
                out = original(*args, **kwargs)
            finally:
                depth[0] -= 1
            if observe is not None:
                observe(index, out)
            return out

        setattr(holder, attr, wrapper)

    try:
        for op in _LAYERS:
            cls = getattr(nn, op, None)
            if cls is not None:
                wrap_layer(cls, op, "layer")
        wrap_layer(nn.Embed, _ATTEND, "layer", method="attend")
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

#: Parameter names of the weights the layers in ``_LAYERS`` own.
_WEIGHT_NAMES = frozenset(name for _, name in _LAYERS.values())


def _trace_id(index: int, op: str, path: tuple[str, ...]) -> str:
    """Node id of a traced call: its parameter path, or its op where it has none."""
    base = "/".join(path) if path else f"{op}_{index}"
    # attend() runs under the same scope as the lookup, so the path alone would collide.
    return f"{base}/attend" if op == _ATTEND else base


def _inferred_id(name: str, index: int) -> str:
    """Node id of an activation identified from values, by the layer it feeds."""
    return f"{name}_before_{'output' if index == _MODEL_OUTPUT else index}"


def _first(result: Any) -> Any:
    """The model output, unwrapping the ``(output, state)`` pair a mutable apply returns."""
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[0]
    return result


def _close(a: np.ndarray, b: np.ndarray) -> bool:
    """Equality to float32 working precision, relative to the magnitude of the values.

    Purely relative, so that a faint signal is compared as finely as a strong one: with an
    absolute floor, ``relu(z)`` and ``z`` agree once ``z`` is small enough.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    scale = float(np.abs(b).max(initial=0.0))
    return bool(np.allclose(a, b, rtol=1e-5, atol=1e-6 * scale))


def _scalar_ratio(base: np.ndarray, x: np.ndarray) -> float | None:
    """``c`` when ``x`` is ``c * base`` to working precision, else ``None``."""
    base = np.asarray(base, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    norm = float(base @ base)
    if base.shape != x.shape or norm == 0.0:
        return None
    factor = float(base @ x) / norm
    return factor if _close(factor * base, x) else None


def _scale_node(index: int, factor: float, previous: str) -> Node:
    """A constant factor read off the probe, applied before the layer at ``index``."""
    return Node(
        _inferred_id("scale", index),
        NodeKind.SCALE,
        "scale",
        (previous,),
        meta={"factor": MomentState.of_values(factor), "input_of": index},
    )


def _holder_of(fn: Any) -> Any:
    """Return the module a registered function lives in, so it can be patched in place."""
    import sys

    module_name = getattr(fn, "__module__", None)
    return sys.modules.get(module_name) if module_name else None


# ------------------------------------------------------------ tied tables

#: Trace op of ``nn.Embed.attend``, the embedding's table used as the output layer.
_ATTEND = "Embed.attend"


def _tied_readout(
    nid: str,
    path: tuple[str, ...],
    previous: str,
    index: int,
    flat: dict[tuple[str, ...], np.ndarray],
) -> Node:
    """``Embed.attend(h)``, which is ``h @ table.T``: a dense layer on the lookup's table."""
    table = (*path, "embedding")
    return Node(
        nid,
        NodeKind.PARAMETRIC,
        _ATTEND,
        (previous,),
        spec=fanmod.ParamSpec(fanmod.DENSE, tuple(int(s) for s in flat[table].shape)),
        # Stored (vocab, d_model), which is already (fan_out, fan_in): no transpose.
        handle=FlaxHandle(table, fanmod.EMBEDDING),
        meta={"path": nid, "trace_index": index},
    )


def _mark_tied(nodes: list[Node]) -> list[Node]:
    """Key the lookup and the ``attend`` of one table alike, for :mod:`anyinit.core.tying`."""
    attended = {
        node.handle.path
        for node in nodes
        if node.op == _ATTEND and isinstance(node.handle, FlaxHandle)
    }
    return [
        node.with_meta(tied="/".join(node.handle.path))
        if isinstance(node.handle, FlaxHandle) and node.handle.path in attended
        else node
        for node in nodes
    ]


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
