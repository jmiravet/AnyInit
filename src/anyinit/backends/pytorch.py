"""PyTorch adapter.

Traces through ``torch.fx``, which yields a full DAG with residual additions and
concatenations as nodes.  When symbolic tracing fails -- data-dependent control flow, or
modules it cannot handle -- the backend falls back to an ordered chain from
``named_modules`` and marks the graph as linear.
"""

from __future__ import annotations

import math
import operator
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core import fan as fanmod
from ..core.graph import FIDELITY_GRAPH, FIDELITY_LINEAR, ModelGraph, Node, NodeKind
from ..core.moments import MomentState
from ..core.registry import ActivationRef
from ..errors import TraceError
from .base import Backend, TapRecorder, module_roots

#: Normalization layer class names, matched by name so a class missing from an older torch
#: release is simply absent.
_NORM_NAMES = frozenset(
    {
        "BatchNorm1d",
        "BatchNorm2d",
        "BatchNorm3d",
        "SyncBatchNorm",
        "LayerNorm",
        "RMSNorm",
        "GroupNorm",
        "InstanceNorm1d",
        "InstanceNorm2d",
        "InstanceNorm3d",
        "LocalResponseNorm",
    }
)

#: Activation module class name -> canonical activation name.
_ACTIVATION_MODULES: dict[str, str] = {
    "ReLU": "relu",
    "ReLU6": "relu6",
    "LeakyReLU": "leaky_relu",
    "RReLU": "leaky_relu",
    "ELU": "elu",
    "SELU": "selu",
    "CELU": "celu",
    "GELU": "gelu",
    "Sigmoid": "sigmoid",
    "SiLU": "silu",
    "Mish": "mish",
    "Softplus": "softplus",
    "Softsign": "softsign",
    "Tanh": "tanh",
    "Hardtanh": "hardtanh",
    "Hardsigmoid": "hardsigmoid",
    "Hardswish": "hardswish",
    "Softmax": "softmax",
    "LogSoftmax": "log_softmax",
}

#: Functional name -> canonical activation name, keyed by ``__name__`` so ``torch.relu``
#: and ``torch.nn.functional.relu`` resolve alike.
_ACTIVATION_FUNCTIONS: dict[str, str] = {
    "relu": "relu",
    "relu6": "relu6",
    "leaky_relu": "leaky_relu",
    "elu": "elu",
    "selu": "selu",
    "celu": "celu",
    "gelu": "gelu",
    "sigmoid": "sigmoid",
    "silu": "silu",
    "mish": "mish",
    "softplus": "softplus",
    "softsign": "softsign",
    "tanh": "tanh",
    "hardtanh": "hardtanh",
    "hardsigmoid": "hardsigmoid",
    "hardswish": "hardswish",
    "softmax": "softmax",
    "log_softmax": "log_softmax",
}

_MERGE_FUNCTIONS = {
    operator.add: "add",
    operator.iadd: "add",
    operator.mul: "mul",
    operator.imul: "mul",
}
_MERGE_NAMES = {"add": "add", "cat": "cat", "concat": "cat", "stack": "cat", "mul": "mul"}
_SHAPE_NAMES = {
    "flatten",
    "reshape",
    "view",
    "permute",
    "transpose",
    "contiguous",
    "squeeze",
    "unsqueeze",
    "size",
    "getattr",
    "getitem",
    "to",
    "detach",
}
_POOL_PREFIXES = ("MaxPool", "AvgPool", "AdaptiveAvgPool", "AdaptiveMaxPool")


@dataclass(frozen=True)
class TorchHandle:
    """Which tensor of which module a scalable node owns."""

    module: Any
    role: str  # weight | in_proj | q_proj | k_proj | v_proj | out_proj | gain

    @property
    def tensor(self) -> Any:
        """The weight tensor this handle addresses."""
        if self.role == "out_proj":
            return self.module.out_proj.weight
        if self.role in _ATTENTION_ROLES:
            return getattr(self.module, f"{self.role}_weight")
        return self.module.weight

    @property
    def bias(self) -> Any:
        """The bias tensor this handle addresses, or ``None``."""
        if self.role == "out_proj":
            return getattr(self.module.out_proj, "bias", None)
        if self.role in ("in_proj", "q_proj"):
            # Separate q/k/v projections still share one packed bias; the query handle
            # owns it so that it is written exactly once.
            return getattr(self.module, "in_proj_bias", None)
        if self.role in _ATTENTION_ROLES:
            return None
        return getattr(self.module, "bias", None)


_ATTENTION_ROLES = frozenset({"in_proj", "q_proj", "k_proj", "v_proj"})


class _Builder:
    """Accumulates the IR nodes of one expanded transformer layer."""

    def __init__(self, nid: str, path: str, layer: Any) -> None:
        self.nid, self.path, self.layer = nid, path, layer
        self.nodes: list[Node] = []

    def _id(self, name: str) -> str:
        return f"{self.nid}_{name}"

    def _push(self, node: Node) -> str:
        self.nodes.append(node)
        return node.id

    def module(self, name: str, source: str) -> str:
        """A parametric or normalization submodule, measured at its output."""
        sub = getattr(self.layer, name)
        backend = TorchBackend.__new__(TorchBackend)
        built = backend._from_module(self._id(name), f"{self.path}.{name}", sub, (source,), ())
        built[-1] = built[-1].with_meta(tap=("out", sub))
        for node in built:
            self._push(node)
        return built[-1].id

    def attention(self, name: str, source: str, memory: str | None = None) -> str:
        """An attention block; its output is the attention module's first output."""
        sub = getattr(self.layer, name)
        backend = TorchBackend.__new__(TorchBackend)
        built = backend._from_attention(
            self._id(name),
            f"{self.path}.{name}",
            sub,
            (source,),
            f"module:{id(sub)}",
            memory=(memory,) if memory else None,
        )
        for node in built:
            self._push(node)
        return built[-1].id

    def dropout(self, name: str, source: str) -> str:
        sub = getattr(self.layer, name)
        return self._push(
            Node(
                self._id(name),
                NodeKind.DROPOUT,
                "Dropout",
                (source,),
                meta={"p": float(getattr(sub, "p", 0.0)), "tap": ("out", sub)},
            )
        )

    def add(self, name: str, a: str, b: str, tap: tuple[str, Any]) -> str:
        return self._push(Node(self._id(name), NodeKind.MERGE, "add", (a, b), meta={"tap": tap}))

    def feed_forward(self, source: str, dropout: str = "dropout2") -> str:
        """``linear2(dropout(activation(linear1(x))))`` followed by its output dropout."""
        h = self.module("linear1", source)
        canonical, params = _transformer_activation(self.layer.activation)
        inner = self.layer.dropout
        h = self._push(
            Node(
                self._id("activation"),
                NodeKind.ACTIVATION,
                canonical,
                (h,),
                meta={"activation": ActivationRef.of(canonical, **params), "tap": ("in", inner)},
            )
        )
        h = self._push(
            Node(
                self._id("dropout"),
                NodeKind.DROPOUT,
                "Dropout",
                (h,),
                meta={"p": float(getattr(inner, "p", 0.0)), "tap": ("in", self.layer.linear2)},
            )
        )
        h = self.module("linear2", h)
        return self.dropout(dropout, h)


_RECURRENT = frozenset({"RNN", "LSTM", "GRU", "RNNCell", "LSTMCell", "GRUCell"})


def _unscaled_reason(owner: Any, model: Any) -> str:
    """Why a parameter was left as the framework initialized it."""
    cls = type(owner).__name__ if owner is not None else ""
    if cls in _RECURRENT:
        return f"recurrent layer ({cls}): recurrence over time is not modeled"
    if owner is model or owner is None:
        return "free parameter used directly in forward(), not through a layer"
    return f"inside {cls}, which AnyInit cannot trace into"


def _transformer_activation(activation: Any) -> tuple[str, dict[str, float]]:
    """Canonical name of a transformer layer's activation, which may be a function."""
    cls = type(activation).__name__
    if cls in _ACTIVATION_MODULES:
        canonical = _ACTIVATION_MODULES[cls]
        return canonical, _module_activation_params(canonical, activation)
    name = getattr(activation, "__name__", "")
    return _ACTIVATION_FUNCTIONS.get(name, name or "relu"), {}


class TorchBackend(Backend):
    """AnyInit adapter for ``torch.nn.Module``."""

    name = "pytorch"
    frameworks = ("torch",)
    requires = "torch"
    install = "torch"

    def __init__(self) -> None:
        import torch  # Imported here, never at module scope: see backends/__init__.

        self._torch = torch
        self._graph_module: Any = None
        self._node_names: dict[str, str] = {}
        self._module_taps: dict[str, tuple[str, Any]] = {}
        self._device: Any = None
        self._dtype: Any = None
        self.last_counts: dict[str, int] = {}

    @staticmethod
    def handles(model: Any) -> bool:
        """Recognize a torch module, declining the ones Keras owns."""
        roots = module_roots(model)
        if roots & {"keras", "tf_keras"}:
            # Keras 3 on its torch backend subclasses nn.Module; that model belongs to
            # the Keras adapter, which knows its layouts and fused activations.
            return False
        return "torch" in roots and hasattr(model, "state_dict")

    def begin(self, model: Any, params: Any = None) -> None:
        """Note where the model lives, so synthesized batches are created to match."""
        first = next((p for p in model.parameters() if p.is_floating_point()), None)
        if first is not None:
            self._device, self._dtype = first.device, first.dtype

    # ----------------------------------------------------------------- graph

    def build_graph(self, model: Any, input_spec: Any = None) -> ModelGraph:
        """Trace through FX, falling back to module order if that fails."""
        try:
            return self._fx_graph(model, input_spec)
        except TraceError:
            raise
        except Exception as exc:
            return self._linear_graph(model, reason=str(exc).splitlines()[0][:160])

    def _fx_graph(self, model: Any, input_spec: Any = None) -> ModelGraph:
        from torch import fx

        leaf_types = tuple(t for t in self.registered_types() if isinstance(t, type))
        native_calls = self.registered_callables()
        unwrappable = _mark_wrapped(native_calls)

        class Tracer(fx.Tracer):
            def is_leaf_module(self, m: Any, qualname: str) -> bool:
                # A registered activation module must stay atomic, otherwise FX inlines
                # it into primitives and the graph loses the fact that it is one
                # activation with one profile.
                if leaf_types and isinstance(m, leaf_types):
                    return True
                return super().is_leaf_module(m, qualname)

        notes = tuple(
            f"activation {name!r} is a nested or dynamically built function, so "
            "torch.fx cannot keep it atomic and will trace into it; define it at module "
            "level, or register an nn.Module subclass instead"
            for name in unwrappable
        )
        graph_module = fx.GraphModule(model, Tracer().trace(model))
        self._graph_module = graph_module
        nodes: list[Node] = []
        emitted: dict[str, str] = {}

        for fx_node in graph_module.graph.nodes:
            inputs = tuple(emitted[n.name] for n in fx_node.all_input_nodes if n.name in emitted)
            readout = _tied_readout(fx_node, model, graph_module, emitted)
            built = (
                [readout]
                if readout is not None
                else self._classify(fx_node, graph_module, inputs, native_calls, leaf_types)
            )
            nodes.extend(built)
            emitted[fx_node.name] = (
                built[-1].id if built else (inputs[0] if inputs else fx_node.name)
            )
            self._node_names[fx_node.name] = emitted[fx_node.name]
        nodes = _mark_tied(nodes)

        # Nodes expanded out of a container module have no FX node of their own; they are
        # read through hooks on the submodule that produces or consumes them.
        self._module_taps = {n.id: n.meta["tap"] for n in nodes if "tap" in n.meta}
        if any(n.kind is NodeKind.POOL and "adaptive" in n.op.lower() for n in nodes):
            shapes = self._probe_shapes(graph_module, input_spec)
            nodes = _with_adaptive_windows(nodes, graph_module, shapes)
        return ModelGraph(nodes, fidelity=FIDELITY_GRAPH, notes=notes)

    def _probe_shapes(self, graph_module: Any, input_spec: Any) -> dict[str, tuple[int, ...]]:
        """Every FX node's output shape, from a one-row batch of zeros.

        Adaptive pooling needs it, since its window is the input's spatial size, which the
        module does not record.  Best effort: an empty result leaves those windows unknown.

        A real pass rather than fake or meta tensors: both route operators through
        ``torch._dynamo`` and with it Triton, whose native library crashes the process when
        TensorFlow was loaded first.  Run only when the model has an adaptive pool.
        """
        torch = self._torch
        from torch import fx

        shapes: dict[str, tuple[int, ...]] = {}

        class ShapeRecorder(fx.Interpreter):
            def run_node(self, n: Any) -> Any:
                out = super().run_node(n)
                if isinstance(out, torch.Tensor):
                    shapes[n.name] = tuple(out.shape)
                return out

        try:
            specs = input_spec if isinstance(input_spec, list) else [input_spec]
            args = []
            for spec in specs:
                if isinstance(spec, torch.Tensor):
                    args.append(torch.zeros_like(spec[:1], device=self._device))
                elif isinstance(spec, tuple) and spec and all(isinstance(d, int) for d in spec):
                    args.append(torch.zeros((1, *spec[1:]), dtype=self._dtype, device=self._device))
                else:
                    return {}
            # Eval mode: a training-mode pass would count a batch on every BatchNorm.
            with _preserved_running_stats(graph_module, training=False), torch.no_grad():
                _quiet(ShapeRecorder(graph_module)).run(*args)
        except Exception:
            return {}
        return shapes

    def _classify(
        self,
        fx_node: Any,
        graph_module: Any,
        inputs: tuple[str, ...],
        native_calls: dict[Any, str],
        leaf_types: tuple[type, ...],
    ) -> list[Node]:
        """Turn one FX node into one or more IR nodes."""
        nid = fx_node.name

        if fx_node.op == "placeholder":
            return [Node(nid, NodeKind.INPUT, "input")]
        if fx_node.op == "output":
            return [Node(nid, NodeKind.OUTPUT, "output", inputs)]
        if fx_node.op == "get_attr":
            return [Node(nid, NodeKind.OTHER, "get_attr", inputs)]

        if fx_node.op == "call_module":
            module = graph_module.get_submodule(str(fx_node.target))
            return self._from_module(nid, str(fx_node.target), module, inputs, leaf_types)

        target = fx_node.target
        name = getattr(target, "__name__", str(target))

        custom = native_calls.get(target)
        if custom is not None:
            return [
                Node(
                    nid,
                    NodeKind.ACTIVATION,
                    custom,
                    inputs,
                    meta={"activation": ActivationRef(custom)},
                )
            ]

        if fx_node.op == "call_function" and target in _MERGE_FUNCTIONS:
            op = _MERGE_FUNCTIONS[target]
            kind = NodeKind.MERGE if len(inputs) > 1 else NodeKind.SHAPE
            return [Node(nid, kind, op, inputs)]

        canonical = _ACTIVATION_FUNCTIONS.get(name)
        if canonical is not None:
            params = _function_activation_params(canonical, fx_node)
            return [
                Node(
                    nid,
                    NodeKind.ACTIVATION,
                    canonical,
                    inputs,
                    meta={"activation": ActivationRef.of(canonical, **params)},
                )
            ]

        if name in _MERGE_NAMES:
            op = _MERGE_NAMES[name]
            kind = NodeKind.MERGE if len(inputs) > 1 else NodeKind.SHAPE
            return [Node(nid, kind, op, inputs)]

        if name.startswith("dropout"):
            p = _arg(fx_node, 1, "p", 0.5)
            return [Node(nid, NodeKind.DROPOUT, name, inputs, meta={"p": float(p)})]

        if "pool" in name:
            window = _pool_window(_arg(fx_node, 1, "kernel_size", 1), _spatial_rank(name))
            pool = "max" if "max" in name else "avg"
            return [Node(nid, NodeKind.POOL, name, inputs, meta={"pool": pool, "window": window})]

        if name in _SHAPE_NAMES or fx_node.op == "call_method":
            return [Node(nid, NodeKind.SHAPE, name, inputs)]

        return [Node(nid, NodeKind.OTHER, name, inputs)]

    def _from_module(
        self,
        nid: str,
        path: str,
        module: Any,
        inputs: tuple[str, ...],
        leaf_types: tuple[type, ...],
    ) -> list[Node]:
        cls = type(module).__name__
        shared = f"module:{id(module)}"

        registered = self.registered_types().get(type(module))
        if registered is not None:
            return [
                Node(
                    nid,
                    NodeKind.ACTIVATION,
                    registered,
                    inputs,
                    meta={"activation": ActivationRef(registered), "path": path},
                )
            ]

        if cls in _ACTIVATION_MODULES:
            canonical = _ACTIVATION_MODULES[cls]
            params = _module_activation_params(canonical, module)
            return [
                Node(
                    nid,
                    NodeKind.ACTIVATION,
                    canonical,
                    inputs,
                    meta={"activation": ActivationRef.of(canonical, **params), "path": path},
                )
            ]

        if cls in _NORM_NAMES:
            if getattr(module, "weight", None) is None:
                return [
                    Node(
                        nid,
                        NodeKind.NORMALIZATION,
                        cls,
                        inputs,
                        meta={"path": path, "affine": False},
                    )
                ]
            return [
                Node(
                    nid,
                    NodeKind.NORMALIZATION,
                    cls,
                    inputs,
                    spec=fanmod.ParamSpec(fanmod.NORM, tuple(module.weight.shape)),
                    handle=TorchHandle(module, "gain"),
                    meta={"path": path, "affine": True, "shared_key": shared},
                )
            ]

        if cls.startswith("Dropout"):
            return [
                Node(
                    nid,
                    NodeKind.DROPOUT,
                    cls,
                    inputs,
                    meta={"p": float(getattr(module, "p", 0.5)), "path": path},
                )
            ]

        if cls.startswith(_POOL_PREFIXES):
            window = _pool_window(getattr(module, "kernel_size", 1), _spatial_rank(cls))
            pool = "max" if "Max" in cls else "avg"
            return [
                Node(
                    nid,
                    NodeKind.POOL,
                    cls,
                    inputs,
                    meta={"pool": pool, "window": window, "path": path},
                )
            ]

        if cls in ("Flatten", "Unflatten", "Identity"):
            return [Node(nid, NodeKind.SHAPE, cls, inputs, meta={"path": path})]

        if cls == "MultiheadAttention":
            return self._from_attention(nid, path, module, inputs, shared)

        expanded = self._from_transformer_stack(nid, path, module, inputs)
        if expanded is not None:
            return expanded

        spec = self._param_spec(module)
        if spec is not None:
            return [
                Node(
                    nid,
                    NodeKind.PARAMETRIC,
                    cls,
                    inputs,
                    spec=spec,
                    handle=TorchHandle(module, "weight"),
                    meta={"path": path, "shared_key": shared},
                )
            ]

        return [Node(nid, NodeKind.OTHER, cls, inputs, meta={"path": path})]

    def _from_attention(
        self,
        nid: str,
        path: str,
        module: Any,
        inputs: tuple[str, ...],
        shared: str,
        memory: tuple[str, ...] | None = None,
    ) -> list[Node]:
        """Linearize attention into its weight blocks.

        Query, key and value share one packed ``(3E, E)`` tensor; its fan_in is ``E`` either
        way, so it scales as one layer.  When the projections are separate
        (``kdim``/``vdim`` differ from the embedding size) each gets its own node, and the
        output projection follows the value path, which is what sets its magnitude.  The
        softmax between them is not pointwise, so it has no profile and is noted.

        ``memory`` is where keys and values come from in cross-attention.
        """
        nodes: list[Node] = []
        kv_source = memory or inputs
        if getattr(module, "in_proj_weight", None) is not None:
            rows, cols = module.in_proj_weight.shape
            nodes.append(
                Node(
                    f"{nid}_in_proj",
                    NodeKind.PARAMETRIC,
                    "MultiheadAttention.in_proj",
                    kv_source,
                    spec=fanmod.ParamSpec(fanmod.ATTENTION, (rows, cols)),
                    handle=TorchHandle(module, "in_proj"),
                    meta={"path": f"{path}.in_proj_weight", "shared_key": f"{shared}:in"},
                )
            )
            value = nodes[-1].id
        else:
            for role, source in (("q_proj", inputs), ("k_proj", kv_source), ("v_proj", kv_source)):
                weight = getattr(module, f"{role}_weight")
                nodes.append(
                    Node(
                        f"{nid}_{role}",
                        NodeKind.PARAMETRIC,
                        f"MultiheadAttention.{role}",
                        source,
                        spec=fanmod.ParamSpec(fanmod.ATTENTION, tuple(weight.shape)),
                        handle=TorchHandle(module, role),
                        meta={"path": f"{path}.{role}_weight", "shared_key": f"{shared}:{role}"},
                    )
                )
            value = nodes[-1].id
        nodes.append(
            Node(
                f"{nid}_out_proj",
                NodeKind.PARAMETRIC,
                "MultiheadAttention.out_proj",
                (value,),
                spec=fanmod.ParamSpec(fanmod.ATTENTION, tuple(module.out_proj.weight.shape)),
                handle=TorchHandle(module, "out_proj"),
                meta={
                    "path": f"{path}.out_proj.weight",
                    "shared_key": f"{shared}:out",
                    "note": "softmax is not pointwise and is not modeled",
                    "tap": ("out", module),
                },
            )
        )
        return nodes

    # --------------------------------------------------- transformer expansion

    def _from_encoder_layer(self, nid: str, path: str, layer: Any, x: str) -> list[Node]:
        """Expand ``nn.TransformerEncoderLayer`` into its real residual structure.

        FX treats every ``torch.nn`` module as a leaf, so without this the whole layer is
        one opaque node and none of its weights are seen.  Both ``norm_first`` layouts are
        covered.  Each node records which submodule's input or output carries its value,
        so the empirical mode can measure it with hooks.
        """
        build = _Builder(nid, path, layer)
        if layer.norm_first:
            h = build.module("norm1", x)
            h = build.attention("self_attn", h)
            h = build.dropout("dropout1", h)
            res = build.add("add1", x, h, tap=("in", layer.norm2))
            h = build.module("norm2", res)
            h = build.feed_forward(h)
            build.add("add2", res, h, tap=("out", layer))
        else:
            h = build.attention("self_attn", x)
            h = build.dropout("dropout1", h)
            res = build.add("add1", x, h, tap=("in", layer.norm1))
            res = build.module("norm1", res)
            h = build.feed_forward(res)
            res = build.add("add2", res, h, tap=("in", layer.norm2))
            build.module("norm2", res)
        return build.nodes

    def _from_decoder_layer(
        self, nid: str, path: str, layer: Any, x: str, memory: str | None
    ) -> list[Node]:
        """Expand ``nn.TransformerDecoderLayer``: self-attention, cross-attention, FFN."""
        build = _Builder(nid, path, layer)
        mem = memory or x
        if layer.norm_first:
            h = build.attention("self_attn", build.module("norm1", x))
            res = build.add("add1", x, build.dropout("dropout1", h), tap=("in", layer.norm2))
            h = build.attention("multihead_attn", build.module("norm2", res), memory=mem)
            res = build.add("add2", res, build.dropout("dropout2", h), tap=("in", layer.norm3))
            h = build.feed_forward(build.module("norm3", res), dropout="dropout3")
            build.add("add3", res, h, tap=("out", layer))
        else:
            h = build.dropout("dropout1", build.attention("self_attn", x))
            res = build.module("norm1", build.add("add1", x, h, tap=("in", layer.norm1)))
            h = build.dropout("dropout2", build.attention("multihead_attn", res, memory=mem))
            res = build.module("norm2", build.add("add2", res, h, tap=("in", layer.norm2)))
            h = build.feed_forward(res, dropout="dropout3")
            build.module("norm3", build.add("add3", res, h, tap=("in", layer.norm3)))
        return build.nodes

    def _from_transformer_stack(
        self, nid: str, path: str, module: Any, inputs: tuple[str, ...]
    ) -> list[Node] | None:
        """Expand the transformer containers FX leaves opaque, or ``None`` if not one."""
        cls = type(module).__name__
        if not inputs:
            return None
        x = inputs[0]
        memory = inputs[1] if len(inputs) > 1 else None
        if cls == "TransformerEncoderLayer":
            return self._from_encoder_layer(nid, path, module, x)
        if cls == "TransformerDecoderLayer":
            return self._from_decoder_layer(nid, path, module, x, memory)
        if cls in ("TransformerEncoder", "TransformerDecoder"):
            nodes: list[Node] = []
            current = x
            for index, layer in enumerate(module.layers):
                sub_id, sub_path = f"{nid}_layers_{index}", f"{path}.layers.{index}"
                if cls == "TransformerEncoder":
                    built = self._from_encoder_layer(sub_id, sub_path, layer, current)
                else:
                    built = self._from_decoder_layer(sub_id, sub_path, layer, current, memory)
                nodes.extend(built)
                current = built[-1].id
            if getattr(module, "norm", None) is not None:
                nodes.extend(
                    self._from_module(f"{nid}_norm", f"{path}.norm", module.norm, (current,), ())
                )
            nodes[-1] = nodes[-1].with_meta(tap=("out", module))
            return nodes
        if cls == "Transformer" and memory is not None:
            encoded = self._from_transformer_stack(
                f"{nid}_encoder", f"{path}.encoder", module.encoder, (x,)
            )
            if not encoded:
                return None
            decoded = self._from_transformer_stack(
                f"{nid}_decoder", f"{path}.decoder", module.decoder, (memory, encoded[-1].id)
            )
            if not decoded:
                return None
            return [*encoded, *decoded]
        return None

    def _param_spec(self, module: Any) -> fanmod.ParamSpec | None:
        """Canonical parameter description for a parametric module."""
        cls = type(module).__name__
        weight = getattr(module, "weight", None)
        if weight is None or not getattr(weight, "requires_grad", False) or weight.ndim < 2:
            return None
        shape = tuple(int(s) for s in weight.shape)
        groups = int(getattr(module, "groups", 1))
        has_bias = getattr(module, "bias", None) is not None

        if cls in {"Linear", "Bilinear"}:
            return fanmod.ParamSpec(fanmod.DENSE, shape, has_bias=has_bias)
        if cls.startswith("Embedding"):
            return fanmod.ParamSpec(fanmod.EMBEDDING, shape)
        if cls.startswith("ConvTranspose"):
            # Stored as (in, out/groups, *kernel): the axes are reversed relative to an
            # ordinary convolution.
            in_ch, out_per_group, *kernel = shape
            canonical = (out_per_group * groups, in_ch // groups, *kernel)
            return fanmod.ParamSpec(
                fanmod.CONV_TRANSPOSE, canonical, groups=groups, has_bias=has_bias
            )
        if cls.startswith("Conv"):
            return fanmod.ParamSpec(fanmod.CONV, shape, groups=groups, has_bias=has_bias)
        return fanmod.ParamSpec(fanmod.DENSE, shape, has_bias=has_bias)

    def _linear_graph(self, model: Any, reason: str) -> ModelGraph:
        """Ordered chain from ``named_modules``, for models FX cannot trace."""
        nodes: list[Node] = [Node("input", NodeKind.INPUT, "input")]
        previous = "input"
        seen: set[int] = set()

        for path, module in model.named_modules():
            if not path or id(module) in seen:
                continue
            if next(module.children(), None) is not None:
                continue  # Containers contribute nothing; their leaves do.
            seen.add(id(module))
            nid = path.replace(".", "_")
            built = self._from_module(nid, path, module, (previous,), ())
            nodes.extend(built)
            previous = built[-1].id

        nodes.append(Node("output", NodeKind.OUTPUT, "output", (previous,)))
        return ModelGraph(
            _mark_tied(nodes),
            fidelity=FIDELITY_LINEAR,
            notes=(
                f"torch.fx could not trace this model ({reason}); fell back to module "
                "registration order, so branches and residual connections are invisible",
            ),
        )

    def unscaled_weights(self, model: Any, graph: ModelGraph) -> list[tuple[str, str]]:
        """Every ``>= 2``-d parameter no graph node writes, with the reason."""
        covered = {
            id(node.handle.tensor)
            for node in graph.scalable
            if isinstance(node.handle, TorchHandle) and node.handle.tensor is not None
        }
        owners = {
            id(param): (module_path, module)
            for module_path, module in model.named_modules()
            for param in module.parameters(recurse=False)
        }
        missing = []
        for name, param in model.named_parameters():
            if param.ndim < 2 or id(param) in covered:
                continue
            _path, owner = owners.get(id(param), ("", None))
            missing.append((name, _unscaled_reason(owner, model)))
        return missing

    # ------------------------------------------------------- numerics bridge

    def eval_elementwise(self, fn: Any, x: np.ndarray) -> np.ndarray:
        """Apply a torch callable to quadrature abscissas."""
        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
        with torch.no_grad():
            out = fn(tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.detach().to(torch.float64).cpu().numpy()

    # ------------------------------------------------------------ parameters

    def read_weight(self, handle: Any) -> np.ndarray:
        """Read a parameter in canonical layout."""
        tensor = handle.tensor.detach().to(self._torch.float64).cpu().numpy()
        return _to_canonical(handle, tensor)

    def write_weight(self, handle: Any, weight: np.ndarray) -> None:
        """Write a canonical weight into the parameter's own layout."""
        tensor = handle.tensor
        native = _from_canonical(handle, weight, tuple(int(s) for s in tensor.shape))
        with self._torch.no_grad():
            tensor.copy_(self._torch.as_tensor(native, dtype=tensor.dtype, device=tensor.device))

    def write_bias(self, handle: Any, bias: np.ndarray) -> None:
        """Write a bias in place."""
        tensor = handle.bias
        if tensor is None:
            return
        with self._torch.no_grad():
            flat = np.asarray(bias, dtype=np.float64).ravel()[: tensor.numel()]
            tensor.copy_(
                self._torch.as_tensor(flat, dtype=tensor.dtype, device=tensor.device).reshape(
                    tensor.shape
                )
            )

    def write_gain(self, handle: Any, gain: float) -> None:
        """Set a normalization scale, and zero its offset."""
        tensor = handle.tensor
        if tensor is None:
            return
        # Assignment rather than multiplication, so repeated calls are idempotent.
        with self._torch.no_grad():
            tensor.fill_(float(gain))
        bias = handle.bias
        if bias is not None:
            with self._torch.no_grad():
                bias.zero_()

    # -------------------------------------------------------------- measuring

    def forward_taps(
        self, model: Any, inputs: Any, taps: Sequence[str], *, training: bool = True
    ) -> dict[str, MomentState]:
        """Interpret the traced graph, recording moments at the requested nodes.

        Uses ``fx.Interpreter`` rather than module hooks so that function nodes -- a bare
        ``torch.relu`` in a ``forward`` -- can be tapped too.
        """
        torch = self._torch
        wanted = set(taps)
        recorder = TapRecorder()
        batch = inputs if isinstance(inputs, (list, tuple)) else (inputs,)

        if self._graph_module is None:
            return self._hook_taps(model, batch, wanted, training)

        from torch import fx

        reverse: dict[str, str] = {}
        for fx_name, node_id in self._node_names.items():
            reverse.setdefault(node_id, fx_name)
        fx_wanted = {reverse.get(t, t) for t in wanted}

        backend = self
        # Once every reachable tap has been recorded the rest of the network is not run:
        # a layer solved early in the model then costs a fraction of a forward pass.
        pending = {t for t in wanted if t in reverse or t in self._module_taps}

        def record(node_id: str, tensor: Any) -> None:
            _record(recorder, node_id, tensor)
            pending.discard(node_id)

        class Recorder(fx.Interpreter):
            def run_node(self, n: Any) -> Any:
                out = super().run_node(n)
                if n.name in fx_wanted and isinstance(out, torch.Tensor):
                    record(backend._node_names.get(n.name, n.name), out)
                if not pending:
                    raise _AllRecordedError
                return out

        handles = self._tap_hooks(wanted, record)
        try:
            with _preserved_running_stats(model, training), torch.no_grad():
                _quiet(Recorder(self._graph_module)).run(*batch)
        except _AllRecordedError:
            pass
        finally:
            for handle in handles:
                handle.remove()
        self.last_counts = recorder.counts()
        return recorder.result()

    def _tap_hooks(self, wanted: set[str], sink: Callable[[str, Any], None]) -> list[Any]:
        """Hook the submodules that expose the wanted expanded nodes."""
        torch = self._torch
        handles = []
        for node_id in wanted & self._module_taps.keys():
            side, module = self._module_taps[node_id]

            def record(tensor: Any, node_id: str = node_id) -> None:
                if isinstance(tensor, tuple):
                    tensor = tensor[0] if tensor else None
                if isinstance(tensor, torch.Tensor):
                    sink(node_id, tensor)

            if side == "in":
                handles.append(module.register_forward_pre_hook(lambda _m, i, f=record: f(i)))
            else:
                handles.append(module.register_forward_hook(lambda _m, _i, o, f=record: f(o)))
        return handles

    def _hook_taps(
        self, model: Any, batch: Any, wanted: set[str], training: bool = True
    ) -> dict[str, MomentState]:
        """Module-hook fallback, used when there is no traced graph."""
        torch = self._torch
        recorder = TapRecorder()
        handles = []
        by_path = {path.replace(".", "_"): path for path, _ in model.named_modules() if path}

        def make_hook(node_id: str) -> Callable[..., None]:
            def hook(_m: Any, _i: Any, output: Any) -> None:
                if isinstance(output, torch.Tensor):
                    _record(recorder, node_id, output)

            return hook

        for node_id in wanted:
            path = by_path.get(node_id)
            if path is None:
                continue
            module = model.get_submodule(path)
            handles.append(module.register_forward_hook(make_hook(node_id)))

        try:
            with _preserved_running_stats(model, training), torch.no_grad():
                model(*batch)
        finally:
            for handle in handles:
                handle.remove()
        self.last_counts = recorder.counts()
        return recorder.result()

    def input_moments(self, inputs: Any) -> MomentState | None:
        """Read the moments of a torch batch."""
        torch = self._torch
        batch = inputs[0] if isinstance(inputs, (list, tuple)) and inputs else inputs
        if not isinstance(batch, torch.Tensor) or not batch.is_floating_point():
            return None
        return super().input_moments(batch.detach().cpu().numpy())

    def make_inputs(self, input_spec: Any, seed: int | None = None) -> Any:
        """Build a torch batch from a shape, a callable or a tensor."""
        torch = self._torch
        if input_spec is None:
            raise ValueError("input_spec is required to build inputs")
        if callable(input_spec) and not isinstance(input_spec, (tuple, list)):
            return input_spec()
        if isinstance(input_spec, torch.Tensor):
            return input_spec if self._device is None else input_spec.to(self._device)
        if isinstance(input_spec, (tuple, list)) and all(isinstance(d, int) for d in input_spec):
            # Drawn on the CPU, so the batch is the same whatever device the model is on.
            generator = torch.Generator().manual_seed(0 if seed is None else int(seed))
            batch = torch.randn(tuple(input_spec), generator=generator)
            return batch if self._device is None else batch.to(self._device, self._dtype)
        return input_spec


# ------------------------------------------------------------ tied tables


def _tied_readout(
    fx_node: Any, model: Any, graph_module: Any, emitted: dict[str, str]
) -> Node | None:
    """An output layer written as a product with an embedding's table.

    ``F.linear(h, emb.weight)`` and ``h @ emb.weight.T`` use the table without a module of
    their own, so without this they would pass for an opaque function and the table would
    be scaled as a lookup alone.  A ``Linear`` head that shares the table is an ordinary
    layer and needs nothing here; :func:`_mark_tied` finds both.
    """
    from torch import fx

    name = getattr(fx_node.target, "__name__", str(fx_node.target))
    if fx_node.op not in ("call_function", "call_method") or name not in ("linear", "matmul"):
        return None
    if len(fx_node.args) < 2 or not isinstance(fx_node.args[0], fx.Node):
        return None
    source, weight = fx_node.args[0], fx_node.args[1]
    transposed = isinstance(weight, fx.Node) and (
        (weight.op == "call_method" and weight.target in ("t", "transpose"))
        or (weight.op == "call_function" and weight.target is getattr and weight.args[1] == "T")
    )
    if transposed:
        weight = weight.args[0]
    # linear(h, W) computes h @ W.T itself; a matmul has to be given the transpose.
    if transposed != (name == "matmul") or getattr(weight, "op", None) != "get_attr":
        return None
    if source.name not in emitted:
        return None

    table = graph_module
    for part in str(weight.target).split("."):
        table = getattr(table, part)
    owner = next(
        (
            (path, module)
            for path, module in model.named_modules()
            if type(module).__name__.startswith("Embedding") and module.weight is table
        ),
        None,
    )
    if owner is None:
        return None
    path, module = owner
    return Node(
        fx_node.name,
        NodeKind.PARAMETRIC,
        name,
        (emitted[source.name],),
        spec=fanmod.ParamSpec(fanmod.DENSE, tuple(int(s) for s in table.shape)),
        handle=TorchHandle(module, "weight"),
        meta={"path": f"{path}.T"},
    )


def _mark_tied(nodes: list[Node]) -> list[Node]:
    """Key every node reading an embedding's table, when more than one reads it.

    Tying assigns one ``Parameter`` to two modules, so it shows in the tensor, not in the
    module.  :mod:`anyinit.core.tying` decides what the table gets.
    """
    readers: dict[int, list[int]] = {}
    for position, node in enumerate(nodes):
        if node.kind is NodeKind.PARAMETRIC and isinstance(node.handle, TorchHandle):
            readers.setdefault(id(node.handle.tensor), []).append(position)

    marked = list(nodes)
    for key, positions in readers.items():
        group = [nodes[p] for p in positions]
        if len(group) > 1 and any(n.spec and n.spec.kind == fanmod.EMBEDDING for n in group):
            for position in positions:
                marked[position] = nodes[position].with_meta(tied=f"tensor:{key}")
    return marked


# ------------------------------------------------------------------ helpers


@contextmanager
def _preserved_running_stats(model: Any, training: bool) -> Iterator[None]:
    """Run in the requested mode, leaving every running statistic as it was.

    Random state is forked and fixed for the duration, so dropout draws the same masks on
    every measurement -- the solver then sees a deterministic function of the scales --
    and the caller's random stream is left untouched.
    """
    import torch

    was_training = model.training
    devices = sorted({p.device.index or 0 for p in model.parameters() if p.device.type == "cuda"})
    snapshots = []
    for module in model.modules():
        for attr in ("running_mean", "running_var", "num_batches_tracked"):
            tensor = getattr(module, attr, None)
            if tensor is not None and hasattr(tensor, "clone"):
                snapshots.append((tensor, tensor.detach().clone()))
    model.train(training)
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)
            yield
    finally:
        model.train(was_training)
        for tensor, original in snapshots:
            tensor.copy_(original)


def _with_adaptive_windows(
    nodes: list[Node], graph_module: Any, shapes: dict[str, tuple[int, ...]]
) -> list[Node]:
    """Give adaptive pools the window their input shape implies."""
    if not shapes:
        return nodes
    sources = {
        n.name: n.all_input_nodes[0].name for n in graph_module.graph.nodes if n.all_input_nodes
    }
    out = []
    for node in nodes:
        source = sources.get(node.id)
        if node.kind is NodeKind.POOL and "adaptive" in node.op.lower() and source in shapes:
            before, after = shapes[source][2:], shapes.get(node.id, ())[2:]
            if before and len(before) == len(after):
                window = math.prod(before) // max(math.prod(after), 1)
                node = node.with_meta(window=max(int(window), 1))
        out.append(node)
    return out


def _quiet(interpreter: Any) -> Any:
    """Keep an interpreter from logging the exceptions that pass through it.

    The early stop in :meth:`TorchBackend.forward_taps` is an exception, and the logging
    path imports ``torch._inductor`` and with it Triton, whose native library crashes the
    process when TensorFlow was loaded first.
    """
    interpreter.extra_traceback = False
    return interpreter


class _AllRecordedError(Exception):
    """Raised inside a measuring pass once nothing further downstream is needed."""


def _record(recorder: TapRecorder, node_id: str, tensor: Any) -> None:
    """Accumulate a tensor's power sums, computed in float64 on its own device."""
    import torch

    flat = tensor.detach().reshape(-1)
    if flat.device.type == "mps":  # No float64 there.
        flat = flat.cpu()
    flat = flat.to(torch.float64)
    if flat.numel() == 0:
        return
    squares = flat * flat
    sums = torch.stack((flat.sum(), squares.sum(), (squares * squares).sum())).tolist()
    recorder.add_sums(node_id, sums[0], sums[1], sums[2], int(flat.numel()))


def _to_canonical(handle: TorchHandle, tensor: np.ndarray) -> np.ndarray:
    if handle.role != "weight":
        return tensor
    cls = type(handle.module).__name__
    if cls.startswith("ConvTranspose"):
        return _transpose_conv_to_canonical(tensor, int(getattr(handle.module, "groups", 1)))
    return tensor


def _from_canonical(
    handle: TorchHandle, weight: np.ndarray, native_shape: tuple[int, ...]
) -> np.ndarray:
    if handle.role != "weight":
        return weight.reshape(native_shape)
    cls = type(handle.module).__name__
    if cls.startswith("ConvTranspose"):
        return _canonical_to_transpose_conv(
            weight, int(getattr(handle.module, "groups", 1)), native_shape
        )
    return weight.reshape(native_shape)


def _transpose_conv_to_canonical(tensor: np.ndarray, groups: int) -> np.ndarray:
    """``(in, out/g, *k)`` -> ``(out, in/g, *k)``."""
    in_ch, out_per_group, *kernel = tensor.shape
    if groups == 1:
        return np.swapaxes(tensor, 0, 1)
    reshaped = tensor.reshape(groups, in_ch // groups, out_per_group, *kernel)
    moved = np.swapaxes(reshaped, 1, 2)
    return moved.reshape(out_per_group * groups, in_ch // groups, *kernel)


def _canonical_to_transpose_conv(
    weight: np.ndarray, groups: int, native_shape: tuple[int, ...]
) -> np.ndarray:
    """Inverse of :func:`_transpose_conv_to_canonical`."""
    if groups == 1:
        return np.swapaxes(weight, 0, 1).reshape(native_shape)
    out_total, in_per_group, *kernel = weight.shape
    reshaped = weight.reshape(groups, out_total // groups, in_per_group, *kernel)
    moved = np.swapaxes(reshaped, 1, 2)
    return moved.reshape(native_shape)


def _mark_wrapped(functions: dict[Any, str]) -> list[str]:
    """Ask FX to emit one node per registered function instead of inlining it.

    ``torch.fx.wrap`` only wraps functions defined in its caller's module, so the registry
    it writes to is addressed directly here.  The patch replaces a module-level name, so a
    closure holding the same function bypasses it, and it lasts for the life of the
    process.  Functions it cannot reach are returned for the caller to report.
    """
    try:
        from torch.fx._symbolic_trace import _wrapped_fns_to_patch
    except ImportError:  # pragma: no cover - FX internals moved
        return [str(label) for label in functions.values()]

    unwrappable: list[str] = []
    for fn, label in functions.items():
        namespace = getattr(fn, "__globals__", None)
        name = getattr(fn, "__name__", None)
        if namespace is None or name is None or namespace.get(name) is not fn:
            # The patch swaps a module-level name, so a closure cannot be reached.
            unwrappable.append(str(label))
            continue
        _wrapped_fns_to_patch[(id(namespace), name)] = namespace
    return unwrappable


def _arg(fx_node: Any, index: int, keyword: str, default: Any) -> Any:
    if keyword in fx_node.kwargs:
        return fx_node.kwargs[keyword]
    if len(fx_node.args) > index:
        return fx_node.args[index]
    return default


def _pool_window(kernel_size: Any, rank: int = 1) -> int:
    """Return the number of values one pooled output reads.

    A scalar kernel size means the same extent along every spatial axis, so it is raised
    to the layer's rank: ``MaxPool2d(3)`` pools nine values.
    """
    if isinstance(kernel_size, (tuple, list)):
        product = 1
        for k in kernel_size:
            product *= int(k)
        return max(product, 1)
    try:
        return max(int(kernel_size) ** max(rank, 1), 1)
    except (TypeError, ValueError):
        return 1


def _spatial_rank(name: str) -> int:
    """Spatial dimensionality implied by a layer or function name."""
    match = re.search(r"([123])\s*[dD]", name)
    return int(match.group(1)) if match else 2


def _module_activation_params(canonical: str, module: Any) -> dict[str, float]:
    if canonical == "leaky_relu":
        slope = getattr(module, "negative_slope", None)
        if slope is None:
            lower = getattr(module, "lower", 0.125)
            upper = getattr(module, "upper", 1.0 / 3.0)
            slope = 0.5 * (float(lower) + float(upper))
        return {"negative_slope": float(slope)}
    if canonical in ("elu", "celu"):
        return {"alpha": float(getattr(module, "alpha", 1.0))}
    if canonical == "softplus":
        return {"beta": float(getattr(module, "beta", 1.0))}
    if canonical == "hardtanh":
        return {
            "lo": float(getattr(module, "min_val", -1.0)),
            "hi": float(getattr(module, "max_val", 1.0)),
        }
    if canonical == "gelu" and getattr(module, "approximate", "none") == "tanh":
        return {}
    return {}


def _function_activation_params(canonical: str, fx_node: Any) -> dict[str, float]:
    if canonical == "leaky_relu":
        return {"negative_slope": float(_arg(fx_node, 1, "negative_slope", 0.01))}
    if canonical in ("elu", "celu"):
        return {"alpha": float(_arg(fx_node, 1, "alpha", 1.0))}
    if canonical == "softplus":
        return {"beta": float(_arg(fx_node, 1, "beta", 1.0))}
    return {}
