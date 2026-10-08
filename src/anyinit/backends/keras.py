"""TensorFlow adapter, by way of Keras.

Keras 3 is the supported surface and covers TensorFlow.  A functional model exposes its
full graph through ``model.operations`` and each operation's inbound nodes, so branches and
merges need no tracing machinery.

Keras fuses activations into layers, so ``Dense(64, activation="relu")`` is split back into
two IR nodes; otherwise the solver would see a layer with no activation after it.
"""

from __future__ import annotations

import contextlib
import functools
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core import fan as fanmod
from ..core.graph import FIDELITY_GRAPH, FIDELITY_LINEAR, ModelGraph, Node, NodeKind
from ..core.moments import MomentState
from ..core.registry import ActivationRef
from .base import Backend, TapRecorder, input_rng, module_roots

_NORM_CLASSES = frozenset(
    {
        "BatchNormalization",
        "LayerNormalization",
        "GroupNormalization",
        "UnitNormalization",
        "RMSNormalization",
        "SpectralNormalization",
    }
)

#: Keras activation function name -> canonical name.
_ACTIVATION_NAMES: dict[str, str] = {
    "relu": "relu",
    "relu6": "relu6",
    "leaky_relu": "leaky_relu",
    "elu": "elu",
    "selu": "selu",
    "celu": "celu",
    "gelu": "gelu",
    "sigmoid": "sigmoid",
    "silu": "silu",
    "swish": "silu",
    "mish": "mish",
    "softplus": "softplus",
    "softsign": "softsign",
    "tanh": "tanh",
    "hard_tanh": "hardtanh",
    "hard_sigmoid": "hardsigmoid",
    "hard_silu": "hardswish",
    "hard_swish": "hardswish",
    "linear": "identity",
    "exponential": "exponential",
    "softmax": "softmax",
    "log_softmax": "log_softmax",
}

_ACTIVATION_LAYERS: dict[str, str] = {
    "ReLU": "relu",
    "LeakyReLU": "leaky_relu",
    "ELU": "elu",
    "PReLU": "leaky_relu",
    "Softmax": "softmax",
    "Activation": "",
}

_MERGE_LAYERS = {
    "Add": "add",
    "Subtract": "add",
    "Multiply": "mul",
    "Average": "add",
    "Concatenate": "cat",
    "Maximum": "add",
    "Minimum": "add",
}

_SHAPE_LAYERS = {
    "Identity",
    "Flatten",
    "Reshape",
    "Permute",
    "RepeatVector",
    "Cropping1D",
    "Cropping2D",
    "Cropping3D",
    "ZeroPadding1D",
    "ZeroPadding2D",
    "ZeroPadding3D",
    "InputLayer",
    "Lambda",
}


@dataclass(frozen=True)
class KerasHandle:
    """Which variable of which layer a scalable node owns."""

    layer: Any
    role: str  # "kernel" | "gain"

    @property
    def variable(self) -> Any:
        """The weight variable this handle addresses."""
        if self.role == "gain":
            return getattr(self.layer, "gamma", None)
        for name in ("kernel", "embeddings", "depthwise_kernel", "pointwise_kernel"):
            found = getattr(self.layer, name, None)
            if found is not None:
                return found
        return None

    @property
    def bias(self) -> Any:
        """The offset variable this handle addresses, or ``None``."""
        return getattr(self.layer, "beta" if self.role == "gain" else "bias", None)


class KerasBackend(Backend):
    """AnyInit adapter for Keras models and layers."""

    name = "keras"
    frameworks = ("keras", "tensorflow")
    requires = "keras"
    install = "tensorflow"

    def __init__(self) -> None:
        import keras  # Imported here, never at module scope.

        self._keras = keras
        self._layers: dict[str, Any] = {}
        self._order: list[tuple[str, Any, tuple[str, ...]]] = []
        self._input_ids: list[str] = []
        self.last_counts: dict[str, int] = {}

    @property
    def label(self) -> str:
        """``keras[jax]``, ``keras[torch]`` or ``keras[tensorflow]``."""
        return f"{self.name}[{self._keras.backend.backend()}]"

    @staticmethod
    def handles(model: Any) -> bool:
        """Recognize a Keras model or layer from its class hierarchy."""
        roots = module_roots(model)
        if not roots & {"keras", "tensorflow", "tf_keras"}:
            return False
        return hasattr(model, "layers") or hasattr(model, "weights")

    # ----------------------------------------------------------------- graph

    def build_graph(self, model: Any, input_spec: Any = None) -> ModelGraph:
        """Trace from Keras' connectivity records, falling back to layer order."""
        operations = getattr(model, "operations", None) or getattr(model, "layers", [])
        wired = any(getattr(op, "_inbound_nodes", None) for op in operations)
        if wired:
            try:
                return self._functional_graph(model, operations)
            except Exception:
                pass
        return self._sequential_graph(model, operations)

    def _functional_graph(self, model: Any, operations: Sequence[Any]) -> ModelGraph:
        """Graph from Keras' own connectivity records."""
        producer: dict[int, str] = {}
        nodes: list[Node] = []
        self._order = []
        self._input_ids = []

        if not any(type(op).__name__ == "InputLayer" for op in operations):
            # A Sequential model leaves its InputLayer out of ``operations``; its input
            # tensors still mark where the signal enters.
            nodes.append(Node("anyinit_input", NodeKind.INPUT, "input"))
            self._input_ids.append("anyinit_input")
            for tensor in _model_inputs(model):
                producer[id(tensor)] = "anyinit_input"

        waiting: list[tuple[Any, Any]] = []
        for op in operations:
            reverse = _reverse_calls(op)
            calls = [node for node in getattr(op, "_inbound_nodes", []) if node not in reverse]
            inputs = tuple(
                producer[id(tensor)]
                for node in calls
                for tensor in _input_tensors(node)
                if id(tensor) in producer
            )
            built = self._from_layer(op, inputs)
            if reverse:
                built = [node.with_meta(tied=_table_key(op)) for node in built]
            nodes.extend(built)
            last = built[-1].id
            for node in calls:
                for tensor in _output_tensors(node):
                    producer[id(tensor)] = last
            if built[0].kind is NodeKind.INPUT:
                self._input_ids.append(last)
            else:
                self._order.append((last, op, inputs))
            waiting.extend((op, call) for call in reverse)
            _emit_tied_readouts(waiting, producer, nodes, self._order)

        outputs = {n.id for n in nodes} - {src for n in nodes for src in n.inputs}
        terminal = [nid for nid in (n.id for n in nodes) if nid in outputs]
        nodes.append(Node("output", NodeKind.OUTPUT, "output", tuple(terminal[-1:])))
        return ModelGraph(nodes, fidelity=FIDELITY_GRAPH)

    def _sequential_graph(self, model: Any, operations: Sequence[Any]) -> ModelGraph:
        """Ordered chain, for Sequential models and unwired subclassed ones."""
        nodes: list[Node] = [Node("anyinit_input", NodeKind.INPUT, "input")]
        previous = "anyinit_input"
        self._order = []
        self._input_ids = ["anyinit_input"]
        for op in operations:
            if type(op).__name__ == "InputLayer":
                continue
            source = previous
            built = self._from_layer(op, (source,))
            nodes.extend(built)
            previous = built[-1].id
            self._order.append((previous, op, (source,)))
        nodes.append(Node("output", NodeKind.OUTPUT, "output", (previous,)))
        return ModelGraph(
            nodes,
            fidelity=FIDELITY_LINEAR,
            notes=(
                "this Keras model exposes no connectivity records, so AnyInit used layer "
                "order; branches and residual connections are invisible",
            ),
        )

    def _from_layer(self, layer: Any, inputs: tuple[str, ...]) -> list[Node]:
        cls = type(layer).__name__
        nid = str(getattr(layer, "name", cls))
        self._layers[nid] = layer

        registered = self.registered_types().get(type(layer))
        if registered is not None:
            return [
                Node(
                    nid,
                    NodeKind.ACTIVATION,
                    registered,
                    inputs,
                    meta={"activation": ActivationRef(registered), "path": nid},
                )
            ]

        if cls in _ACTIVATION_LAYERS:
            canonical = _ACTIVATION_LAYERS[cls] or _activation_name(layer)
            if canonical == "identity":
                return [Node(nid, NodeKind.SHAPE, cls, inputs, meta={"path": nid})]
            params = _layer_activation_params(canonical, layer)
            return [
                Node(
                    nid,
                    NodeKind.ACTIVATION,
                    canonical,
                    inputs,
                    meta={"activation": ActivationRef.of(canonical, **params), "path": nid},
                )
            ]

        if cls in _MERGE_LAYERS:
            op = _MERGE_LAYERS[cls]
            kind = NodeKind.MERGE if len(inputs) > 1 else NodeKind.SHAPE
            return [Node(nid, kind, op, inputs, meta={"path": nid})]

        if cls in _NORM_CLASSES:
            if getattr(layer, "gamma", None) is None:
                return [
                    Node(
                        nid,
                        NodeKind.NORMALIZATION,
                        cls,
                        inputs,
                        meta={"path": nid, "affine": False},
                    )
                ]
            return [
                Node(
                    nid,
                    NodeKind.NORMALIZATION,
                    cls,
                    inputs,
                    spec=fanmod.ParamSpec(fanmod.NORM, tuple(layer.gamma.shape)),
                    handle=KerasHandle(layer, "gain"),
                    meta={"path": nid, "affine": True},
                )
            ]

        if cls.startswith("Dropout") or cls in (
            "SpatialDropout1D",
            "SpatialDropout2D",
            "SpatialDropout3D",
            "GaussianDropout",
        ):
            return [
                Node(
                    nid,
                    NodeKind.DROPOUT,
                    cls,
                    inputs,
                    meta={"p": float(getattr(layer, "rate", 0.5)), "path": nid},
                )
            ]

        if "Pooling" in cls:
            window = _pool_window(getattr(layer, "pool_size", 1), _spatial_rank(cls))
            pool = "max" if "Max" in cls else "avg"
            return [
                Node(
                    nid,
                    NodeKind.POOL,
                    cls,
                    inputs,
                    meta={"pool": pool, "window": window, "path": nid},
                )
            ]

        if cls in _SHAPE_LAYERS:
            kind = NodeKind.INPUT if cls == "InputLayer" else NodeKind.SHAPE
            return [Node(nid, kind, cls, inputs, meta={"path": nid})]

        spec = self._param_spec(layer)
        if spec is None:
            return [Node(nid, NodeKind.OTHER, cls, inputs, meta={"path": nid})]

        built = [
            Node(
                nid,
                NodeKind.PARAMETRIC,
                cls,
                inputs,
                spec=spec,
                handle=KerasHandle(layer, "kernel"),
                meta={"path": nid},
            )
        ]
        fused = _activation_name(layer)
        if fused and fused != "identity":
            # Split the fused activation out, so the solver can see what this layer feeds.
            built.append(
                Node(
                    f"{nid}_activation",
                    NodeKind.ACTIVATION,
                    fused,
                    (nid,),
                    meta={
                        "activation": ActivationRef(fused),
                        "path": f"{nid}.activation",
                        "fused_into": nid,
                    },
                )
            )
        return built

    def _param_spec(self, layer: Any) -> fanmod.ParamSpec | None:
        cls = type(layer).__name__
        variable = KerasHandle(layer, "kernel").variable
        if variable is None:
            return None
        shape = tuple(int(s) for s in variable.shape)
        if len(shape) < 2:
            return None
        groups = int(getattr(layer, "groups", 1) or 1)
        has_bias = getattr(layer, "bias", None) is not None

        if _is_embedding(layer):
            return fanmod.ParamSpec(fanmod.EMBEDDING, shape)
        if cls in {"Dense", "EinsumDense"}:
            return fanmod.ParamSpec(fanmod.DENSE, (shape[1], shape[0]), has_bias=has_bias)
        if "Transpose" in cls:
            # Keras stores (*kernel, out, in); canonical wants (out, in, *kernel).
            *kernel, out_ch, in_ch = shape
            return fanmod.ParamSpec(
                fanmod.CONV_TRANSPOSE,
                (out_ch, in_ch // groups, *kernel),
                groups=groups,
                has_bias=has_bias,
            )
        if cls.startswith(("Conv", "SeparableConv", "Depthwise")):
            # Keras stores (*kernel, in, out); canonical wants (out, in, *kernel).
            *kernel, in_ch, out_ch = shape
            if cls.startswith("Depthwise"):
                groups = in_ch
                return fanmod.ParamSpec(
                    fanmod.CONV, (in_ch * out_ch, 1, *kernel), groups=groups, has_bias=has_bias
                )
            return fanmod.ParamSpec(
                fanmod.CONV, (out_ch, in_ch, *kernel), groups=groups, has_bias=has_bias
            )
        return fanmod.ParamSpec(fanmod.DENSE, (shape[-1], shape[0]), has_bias=has_bias)

    def unscaled_weights(self, model: Any, graph: ModelGraph) -> list[tuple[str, str]]:
        """Every ``>= 2``-d variable no graph node writes, with the reason."""
        covered = {
            id(node.handle.variable)
            for node in graph.scalable
            if isinstance(node.handle, KerasHandle) and node.handle.variable is not None
        }
        missing = []
        for layer in getattr(model, "layers", [model]):
            for variable in getattr(layer, "weights", []):
                if len(variable.shape) < 2 or id(variable) in covered:
                    continue
                name = getattr(variable, "path", None) or (
                    f"{layer.name}/{getattr(variable, 'name', 'weight')}"
                )
                cls = type(layer).__name__
                reason = (
                    f"recurrent layer ({cls}): recurrence over time is not modeled"
                    if cls in ("LSTM", "GRU", "SimpleRNN", "RNN", "Bidirectional")
                    else f"inside {cls}, which AnyInit has no adapter for"
                )
                missing.append((name, reason))
        return missing

    # ------------------------------------------------------- numerics bridge

    def eval_elementwise(self, fn: Any, x: np.ndarray) -> np.ndarray:
        """Apply a Keras callable to quadrature abscissas."""
        ops = self._keras.ops
        out = fn(ops.convert_to_tensor(np.ascontiguousarray(x, dtype=np.float32)))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return np.asarray(ops.convert_to_numpy(out), dtype=np.float64)

    # ------------------------------------------------------------ parameters

    def read_weight(self, handle: Any) -> np.ndarray:
        """Read a variable, converting Keras' axis order to canonical."""
        variable = handle.variable
        native = np.asarray(self._keras.ops.convert_to_numpy(variable), dtype=np.float64)
        return _to_canonical(handle, native)

    def write_weight(self, handle: Any, weight: np.ndarray) -> None:
        """Write a canonical weight into a variable, in Keras' axis order."""
        variable = handle.variable
        if variable is None:
            return
        native = _from_canonical(handle, weight, tuple(int(s) for s in variable.shape))
        variable.assign(native.astype(_dtype_of(variable), copy=False))

    def write_bias(self, handle: Any, bias: np.ndarray) -> None:
        """Write a bias variable."""
        variable = handle.bias
        if variable is None:
            return
        size = int(np.prod(variable.shape))
        flat = np.asarray(bias, dtype=np.float64).ravel()[:size]
        variable.assign(flat.reshape(variable.shape).astype(_dtype_of(variable), copy=False))

    def write_gain(self, handle: Any, gain: float) -> None:
        """Set a normalization scale, and zero its offset."""
        variable = handle.variable
        if variable is None:
            return
        # Assignment, not multiplication, so repeated calls are idempotent.
        variable.assign(np.full(variable.shape, float(gain), dtype=_dtype_of(variable)))
        bias = handle.bias
        if bias is not None:
            bias.assign(np.zeros(bias.shape, dtype=_dtype_of(bias)))

    # -------------------------------------------------------------- measuring

    def forward_taps(
        self, model: Any, inputs: Any, taps: Sequence[str], *, training: bool = True
    ) -> dict[str, MomentState]:
        """Execute the recorded graph op by op, recording moments at the taps.

        Driving the ops directly keeps one code path for functional and sequential models,
        and reaches the activations split out of fused layers, which a Keras sub-model
        cannot expose as separate outputs.
        """
        ops = self._keras.ops
        wanted = set(taps)
        recorder = TapRecorder()
        tensor = ops.convert_to_tensor(inputs)
        values: dict[str, Any] = dict.fromkeys(self._input_ids, tensor)

        snapshots = _snapshot_moving_stats(self._order, ops)
        with _fixed_global_rng(self._keras):
            self._run_order(values, wanted, recorder, training)
        _restore_moving_stats(snapshots)
        self.last_counts = recorder.counts()
        return recorder.result()

    def _run_order(
        self, values: dict[str, Any], wanted: set[str], recorder: TapRecorder, training: bool
    ) -> None:
        ops = self._keras.ops
        for node_id, layer, input_ids in self._order:
            args = [values[i] for i in input_ids if i in values]
            if not args:
                continue  # An unreachable op: nothing upstream produced its input.
            argument = args if len(args) > 1 else args[0]
            try:
                out = layer(argument, training=training)
            except TypeError:
                try:
                    out = layer(argument)
                except Exception:
                    continue
            except Exception:
                continue
            values[node_id] = out
            if node_id in wanted:
                recorder.add(node_id, ops.convert_to_numpy(out))

    def make_inputs(self, input_spec: Any, seed: int | None = None) -> Any:
        """Build a Keras batch from a shape, a callable or an array."""
        if input_spec is None:
            raise ValueError("input_spec is required to build inputs")
        if callable(input_spec) and not isinstance(input_spec, (tuple, list)):
            return input_spec()
        if isinstance(input_spec, (tuple, list)) and all(isinstance(d, int) for d in input_spec):
            rng = input_rng(seed)
            return rng.standard_normal(tuple(input_spec)).astype(np.float32)
        return input_spec


# ------------------------------------------------------------------ helpers


def _snapshot_moving_stats(order: Sequence[Any], ops: Any) -> list[tuple[Any, np.ndarray]]:
    """Copy out the state a training-mode pass would advance.

    That is the moving statistics, and the seed state behind dropout: restoring the latter
    makes every measurement draw the same masks, so the solver sees a deterministic
    function of the scales.
    """
    saved: list[tuple[Any, np.ndarray]] = []
    for _node_id, layer, _inputs in order:
        variables = [getattr(layer, attr, None) for attr in ("moving_mean", "moving_variance")]
        generator = getattr(layer, "seed_generator", None)
        variables.append(getattr(generator, "state", None))
        saved.extend(
            (variable, np.asarray(ops.convert_to_numpy(variable)))
            for variable in variables
            if variable is not None
        )
    return saved


@contextlib.contextmanager
def _fixed_global_rng(keras: Any) -> Iterator[None]:
    """Fix the global random stream that unseeded layers fall back to.

    Only Keras' torch backend has one: there an unseeded ``Dropout`` draws from torch's
    global generator.  It is forked, so the caller's stream is left untouched.
    """
    if keras.backend.backend() != "torch":
        yield
        return
    import torch

    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        yield


def _restore_moving_stats(saved: Sequence[tuple[Any, np.ndarray]]) -> None:
    for variable, original in saved:
        variable.assign(original)


def _dtype_of(variable: Any) -> Any:
    return np.dtype(getattr(variable, "dtype", "float32"))


def _input_tensors(node: Any) -> Sequence[Any]:
    tensors = getattr(node, "input_tensors", None)
    if tensors is None:
        return ()
    return tensors if isinstance(tensors, (list, tuple)) else (tensors,)


def _model_inputs(model: Any) -> Sequence[Any]:
    try:
        tensors = model.inputs
    except (AttributeError, ValueError):
        return ()
    if tensors is None:
        return ()
    return tensors if isinstance(tensors, (list, tuple)) else (tensors,)


def _output_tensors(node: Any) -> Sequence[Any]:
    tensors = getattr(node, "output_tensors", None)
    if tensors is None:
        return ()
    return tensors if isinstance(tensors, (list, tuple)) else (tensors,)


def _activation_name(layer: Any) -> str:
    """Canonical name of a layer's fused activation, or an empty string."""
    activation = getattr(layer, "activation", None)
    if activation is None:
        return ""
    raw = getattr(activation, "__name__", None) or str(activation)
    return _ACTIVATION_NAMES.get(raw, raw if raw in _ACTIVATION_NAMES.values() else "")


def _layer_activation_params(canonical: str, layer: Any) -> dict[str, float]:
    if canonical == "leaky_relu":
        slope = getattr(layer, "negative_slope", None)
        if slope is None:
            slope = getattr(layer, "alpha", 0.3)
        return {"negative_slope": float(np.mean(np.asarray(slope, dtype=np.float64)))}
    if canonical == "elu":
        return {"alpha": float(getattr(layer, "alpha", 1.0))}
    return {}


def _pool_window(pool_size: Any, rank: int = 1) -> int:
    """Return the number of values one pooled output reads.

    A scalar pool size means the same extent along every spatial axis, so it is raised to
    the layer's rank.
    """
    if isinstance(pool_size, (tuple, list)):
        product = 1
        for k in pool_size:
            product *= int(k)
        return max(product, 1)
    try:
        return max(int(pool_size) ** max(rank, 1), 1)
    except (TypeError, ValueError):
        return 1


def _spatial_rank(name: str) -> int:
    """Spatial dimensionality implied by a layer name."""
    match = re.search(r"([123])\s*[dD]", name)
    return int(match.group(1)) if match else 2


# ------------------------------------------------------------ tied tables


def _is_embedding(layer: Any) -> bool:
    """Whether ``layer`` is an ``Embedding`` or a subclass, such as keras_hub's tied one."""
    return any(cls.__name__ == "Embedding" for cls in type(layer).__mro__)


def _reverse_calls(layer: Any) -> list[Any]:
    """Calls that use an embedding's table as the output layer.

    Keras has no tying of its own; ``keras_hub.layers.ReversibleEmbedding`` is the usual
    way, called a second time as ``layer(h, reverse=True)`` to compute ``h @ table.T``.
    """
    if not _is_embedding(layer):
        return []
    return [
        call
        for call in getattr(layer, "_inbound_nodes", [])
        if getattr(getattr(call, "arguments", None), "kwargs", {}).get("reverse")
    ]


def _table_key(layer: Any) -> str:
    return f"variable:{id(layer.embeddings)}"


def _emit_tied_readouts(
    waiting: list[tuple[Any, Any]],
    producer: dict[int, str],
    nodes: list[Node],
    order: list[tuple[str, Any, tuple[str, ...]]],
) -> None:
    """Emit each waiting ``reverse=True`` call once everything it reads has been emitted.

    A layer appears once among a model's operations, where it is first called, but its
    reverse call reads the end of the network, so it is placed later, as its own node.
    """
    for layer, call in list(waiting):
        sources = tuple(producer.get(id(tensor)) for tensor in _input_tensors(call))
        if not sources or None in sources:
            continue
        waiting.remove((layer, call))
        nid = f"{layer.name}_reverse"
        nodes.append(
            Node(
                nid,
                NodeKind.PARAMETRIC,
                type(layer).__name__,
                tuple(str(s) for s in sources),
                spec=fanmod.ParamSpec(fanmod.DENSE, tuple(int(s) for s in layer.embeddings.shape)),
                # Stored (vocab, d_model), which is already (fan_out, fan_in).
                handle=KerasHandle(layer, "kernel"),
                meta={"path": nid, "tied": _table_key(layer)},
            )
        )
        order.append((nid, functools.partial(layer, reverse=True), nodes[-1].inputs))
        for tensor in _output_tensors(call):
            producer[id(tensor)] = nid


def _to_canonical(handle: KerasHandle, native: np.ndarray) -> np.ndarray:
    if handle.role != "kernel" or native.ndim < 2:
        return native
    cls = type(handle.layer).__name__
    if _is_embedding(handle.layer):
        return native
    if cls in ("Dense", "EinsumDense"):
        return native.T
    if "Transpose" in cls:  # (*kernel, out, in) -> (out, in, *kernel)
        return np.moveaxis(native, (-2, -1), (0, 1))
    if cls.startswith("Depthwise"):  # (*kernel, in, mult) -> (in*mult, 1, *kernel)
        *kernel, in_ch, mult = native.shape
        return np.moveaxis(native, (-2, -1), (0, 1)).reshape(in_ch * mult, 1, *kernel)
    if cls.startswith(("Conv", "SeparableConv")):
        return np.moveaxis(native, (-1, -2), (0, 1))  # (*kernel, in, out) -> (out, in, *kernel)
    return native.T


def _from_canonical(
    handle: KerasHandle, weight: np.ndarray, native_shape: tuple[int, ...]
) -> np.ndarray:
    if handle.role != "kernel" or weight.ndim < 2:
        return weight.reshape(native_shape)
    cls = type(handle.layer).__name__
    if _is_embedding(handle.layer):
        return weight.reshape(native_shape)
    if cls in ("Dense", "EinsumDense"):
        return weight.T.reshape(native_shape)
    if "Transpose" in cls:
        return np.moveaxis(weight, (0, 1), (-2, -1)).reshape(native_shape)
    if cls.startswith("Depthwise"):
        total, _one, *kernel = weight.shape
        mult = native_shape[-1]
        reshaped = weight.reshape(total // mult, mult, *kernel)
        return np.moveaxis(reshaped, (0, 1), (-2, -1)).reshape(native_shape)
    if cls.startswith(("Conv", "SeparableConv")):
        return np.moveaxis(weight, (0, 1), (-1, -2)).reshape(native_shape)
    return weight.T.reshape(native_shape)
