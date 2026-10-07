"""JAX/Flax specifics: functional parameters, instrumented tracing, layouts."""

from __future__ import annotations

import numpy as np
import pytest

import anyinit
from anyinit.core.graph import FIDELITY_LINEAR, NodeKind
from anyinit.errors import TraceError

pytest.importorskip("flax")
jax = pytest.importorskip("jax")
jnp = jax.numpy
import flax.linen as nn  # noqa: E402
from flax.traverse_util import flatten_dict  # noqa: E402


class MLP(nn.Module):
    depth: int = 3
    width: int = 32

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = jax.nn.relu(nn.Dense(self.width)(x))
        return nn.Dense(8)(x)


@pytest.fixture
def mlp():
    model = MLP()
    params = model.init(jax.random.key(0), jnp.ones((1, 16)))
    return model, params


def test_detects_flax_modules(mlp):
    from anyinit.backends import resolve

    model, _params = mlp
    assert resolve(model).name == "jax"


def test_instrumented_trace_recovers_the_call_order(mlp):
    from anyinit.backends import resolve

    model, params = mlp
    backend = resolve(model)
    backend.begin(model, params)
    graph = backend.build_graph(model, (4, 16))

    kinds = [(n.kind, n.op) for n in graph if n.kind is not NodeKind.OUTPUT]
    assert kinds == [
        (NodeKind.INPUT, "input"),
        (NodeKind.PARAMETRIC, "Dense"),
        (NodeKind.ACTIVATION, "relu"),
        (NodeKind.PARAMETRIC, "Dense"),
        (NodeKind.ACTIVATION, "relu"),
        (NodeKind.PARAMETRIC, "Dense"),
        (NodeKind.ACTIVATION, "relu"),
        (NodeKind.PARAMETRIC, "Dense"),
    ]


def test_graph_is_declared_linear(mlp):
    """Check that the graph is reported as linear.

    The instrumentation recovers call order, not structure, so a residual add is
    invisible to it and the report must say so.
    """
    model, params = mlp
    _params, report = anyinit.initialize_params(model, params, (4, 16), seed=0)
    assert report.graph_fidelity == FIDELITY_LINEAR
    assert any("call order, not graph structure" in w for w in report.warnings)


def test_node_ids_are_the_parameter_paths(mlp):
    from anyinit.backends import resolve

    model, params = mlp
    backend = resolve(model)
    backend.begin(model, params)
    graph = backend.build_graph(model, (4, 16))
    names = [n.id for n in graph.of_kind(NodeKind.PARAMETRIC)]
    assert names == ["Dense_0", "Dense_1", "Dense_2", "Dense_3"]


def test_nested_modules_get_their_full_path():
    class Block(nn.Module):
        @nn.compact
        def __call__(self, x):
            return jax.nn.relu(nn.Dense(16)(x))

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(4)(Block()(x))

    from anyinit.backends import resolve

    model = Net()
    params = model.init(jax.random.key(0), jnp.ones((1, 8)))
    backend = resolve(model)
    backend.begin(model, params)
    graph = backend.build_graph(model, (4, 8))
    assert [n.id for n in graph.of_kind(NodeKind.PARAMETRIC)] == ["Block_0/Dense_0", "Dense_0"]


def test_initialization_is_functional(mlp):
    """Check that the input tree is untouched and the result is a new tree.

    JAX parameters are immutable, so initialization has to be functional.
    """
    model, params = mlp
    before = np.asarray(flatten_dict(params["params"])[("Dense_0", "kernel")]).copy()

    new_params, report = anyinit.initialize_params(model, params, (4, 16), seed=0)

    after_original = np.asarray(flatten_dict(params["params"])[("Dense_0", "kernel")])
    after_new = np.asarray(flatten_dict(new_params["params"])[("Dense_0", "kernel")])
    assert np.allclose(before, after_original), "the input tree was mutated"
    assert not np.allclose(before, after_new), "the returned tree was not initialized"
    assert report.params is new_params


def test_returned_tree_has_the_same_structure(mlp):
    model, params = mlp
    new_params, _report = anyinit.initialize_params(model, params, (4, 16), seed=0)
    assert set(flatten_dict(params["params"])) == set(flatten_dict(new_params["params"]))


def test_returned_tree_runs(mlp):
    model, params = mlp
    new_params, _report = anyinit.initialize_params(model, params, (4, 16), seed=0)
    out = model.apply(new_params, jnp.ones((4, 16)))
    assert out.shape == (4, 8)
    assert bool(jnp.isfinite(out).all())


def test_init_without_params_needs_a_shape():
    from anyinit.backends import resolve

    model = MLP()
    backend = resolve(model)
    with pytest.raises(TraceError, match="input_spec"):
        backend.build_graph(model, None)


def test_params_are_discovered_from_input_spec_alone():
    """No parameter tree supplied: the backend calls model.init itself."""
    model = MLP()
    report = anyinit.initialize(model, input_spec=(4, 16), seed=0)
    assert report.params is not None
    assert report.converged


def test_instrumentation_is_fully_restored(mlp):
    model, params = mlp
    original_relu = jax.nn.relu
    original_dense_call = nn.Dense.__call__
    anyinit.initialize_params(model, params, (4, 16), seed=0)
    assert jax.nn.relu is original_relu
    assert nn.Dense.__call__ is original_dense_call


def test_instrumentation_is_restored_even_on_failure():
    original_relu = jax.nn.relu

    class Broken(nn.Module):
        @nn.compact
        def __call__(self, x):
            raise RuntimeError("boom")

    model = Broken()
    with pytest.raises(Exception):
        anyinit.initialize(model, input_spec=(4, 8), seed=0)
    assert jax.nn.relu is original_relu


def test_conv_layout_round_trips():
    class ConvNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Conv(features=8, kernel_size=(3, 3))(x)

    from anyinit.backends import resolve

    model = ConvNet()
    params = model.init(jax.random.key(0), jnp.ones((1, 8, 8, 4)))
    backend = resolve(model)
    backend.begin(model, params)
    graph = backend.build_graph(model, (2, 8, 8, 4))
    node = next(n for n in graph.of_kind(NodeKind.PARAMETRIC))

    assert node.spec.canonical_shape == (8, 4, 3, 3)
    original = np.arange(node.spec.size, dtype=np.float64).reshape(node.spec.canonical_shape)
    backend.write_weight(node.handle, original)
    assert np.allclose(backend.read_weight(node.handle), original)


def test_layernorm_scale_is_assigned():
    class Normed(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(16)(x)
            x = nn.LayerNorm()(x)
            return jax.nn.relu(x)

    model = Normed()
    params = model.init(jax.random.key(0), jnp.ones((1, 16)))
    gains = []
    for _ in range(3):
        params, _report = anyinit.initialize_params(model, params, (32, 16), seed=0)
        gains.append(float(np.asarray(flatten_dict(params["params"])[("LayerNorm_0", "scale")])[0]))
    assert gains[0] == pytest.approx(gains[1]) == pytest.approx(gains[2])


def test_model_with_no_recognised_layers_is_reported():
    class Raw(nn.Module):
        @nn.compact
        def __call__(self, x):
            return x * 2.0

    with pytest.raises(TraceError, match="invisible"):
        anyinit.initialize(Raw(), input_spec=(4, 8), seed=0)
