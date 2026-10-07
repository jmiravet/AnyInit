"""Keras specifics: the functional DAG, fused activations and Keras layouts."""

from __future__ import annotations

import numpy as np
import pytest

import anyinit
from anyinit.core.graph import FIDELITY_GRAPH, NodeKind

keras = pytest.importorskip("keras")
layers = keras.layers


@pytest.fixture
def backend():
    from anyinit.backends import resolve

    return resolve(keras.Sequential([layers.Dense(4)]))


def test_detects_keras_models(backend):
    assert backend.name == "keras"


def test_keras_on_torch_is_not_claimed_by_the_torch_backend(backend):
    """Check that a Keras model is claimed by the Keras adapter.

    Keras 3 on its torch backend subclasses nn.Module, so both adapters match it, and
    the Keras one knows its layouts and fused activations.
    """
    assert backend.name == "keras"


def test_functional_graph_sees_the_merge():
    """Check that both branches of a merge are paired with the activation after it.

    The ResNet transition shape: two normalizations into an add, then one activation.
    """
    inputs = keras.Input((32,))
    skip = layers.BatchNormalization()(layers.Dense(32)(inputs))
    main = layers.BatchNormalization()(layers.Dense(32)(inputs))
    merged = layers.Activation("relu")(layers.Add()([skip, main]))
    model = keras.Model(inputs, layers.Dense(4)(merged))

    from anyinit.backends import resolve
    from anyinit.core.topology import build_plan

    graph = resolve(model).build_graph(model)
    assert graph.fidelity == FIDELITY_GRAPH
    assert [n.op for n in graph.of_kind(NodeKind.MERGE)] == ["add"]

    plan = build_plan(graph)
    relu = next(n.id for n in graph.of_kind(NodeKind.ACTIVATION))
    assert len(plan.ancestors[relu]) == 2, plan.ancestors[relu]


def test_activation_with_no_scalable_input_is_reported():
    """Check that an activation with no adjustable input is reported.

    Two already-activated branches merging into a third leave nothing upstream carrying
    a scale.
    """
    inputs = keras.Input((32,))
    a = layers.Dense(32, activation="relu")(inputs)
    b = layers.Activation("relu")(layers.Dense(32)(a))
    merged = layers.Activation("relu")(layers.Add()([a, b]))
    model = keras.Model(inputs, layers.Dense(4)(merged))

    report = anyinit.initialize(model, input_spec=(128, 32), seed=0)
    assert any("no scalable layer feeding it" in w for w in report.warnings)


def test_fused_activation_becomes_its_own_node(backend):
    model = keras.Sequential([keras.Input((16,)), layers.Dense(8, activation="gelu")])
    graph = backend.build_graph(model)
    assert [n.op for n in graph.of_kind(NodeKind.ACTIVATION)] == ["gelu"]
    assert [n.op for n in graph.of_kind(NodeKind.PARAMETRIC)] == ["Dense"]


def test_layer_without_activation_has_no_activation_node(backend):
    model = keras.Sequential([keras.Input((16,)), layers.Dense(8)])
    graph = backend.build_graph(model)
    assert graph.of_kind(NodeKind.ACTIVATION) == ()


@pytest.mark.parametrize(
    "layer_factory",
    [
        lambda: layers.Dense(8),
        lambda: layers.Conv1D(8, 3),
        lambda: layers.Conv2D(8, 3),
        lambda: layers.Conv3D(8, 3),
        lambda: layers.Conv2DTranspose(8, 3),
        lambda: layers.Embedding(100, 8),
    ],
    ids=["Dense", "Conv1D", "Conv2D", "Conv3D", "Conv2DTranspose", "Embedding"],
)
def test_weight_layout_round_trips(backend, layer_factory):
    shapes = {
        "Dense": (4, 16),
        "Conv1D": (4, 16, 8),
        "Conv2D": (4, 8, 8, 3),
        "Conv3D": (2, 4, 4, 4, 3),
        "Conv2DTranspose": (4, 8, 8, 3),
        "Embedding": (4, 7),
    }
    layer = layer_factory()
    name = type(layer).__name__
    model = keras.Sequential([keras.Input(shapes[name][1:]), layer])
    graph = backend.build_graph(model)
    node = next(n for n in graph.of_kind(NodeKind.PARAMETRIC))

    original = np.arange(node.spec.size, dtype=np.float64).reshape(node.spec.canonical_shape)
    backend.write_weight(node.handle, original)
    assert np.allclose(backend.read_weight(node.handle), original)


def test_normalization_gain_is_assigned():
    model = keras.Sequential(
        [keras.Input((16,)), layers.Dense(16), layers.BatchNormalization(), layers.ReLU()]
    )
    gains = []
    for _ in range(3):
        anyinit.initialize(model, input_spec=(64, 16), seed=0)
        gains.append(float(keras.ops.convert_to_numpy(model.layers[1].gamma)[0]))
    assert gains[0] == pytest.approx(gains[1]) == pytest.approx(gains[2])


def test_validation_leaves_moving_statistics_untouched():
    """Check that validating does not disturb the moving statistics.

    Measuring has to happen in training mode, since at initialization a batch-norm
    layer's running statistics are still (0, 1) and inference mode would not normalize.
    """
    model = keras.Sequential(
        [keras.Input((16,)), layers.Dense(16), layers.BatchNormalization(), layers.ReLU()]
    )
    norm = model.layers[1]
    before = np.array(keras.ops.convert_to_numpy(norm.moving_variance))
    anyinit.initialize(model, input_spec=(128, 16), seed=0)
    after = np.array(keras.ops.convert_to_numpy(norm.moving_variance))
    assert np.allclose(before, after)


def test_sequential_model_is_initialized():
    model = keras.Sequential(
        [keras.Input((32,))] + [layers.Dense(64, activation="relu") for _ in range(4)]
    )
    report = anyinit.initialize(model, input_spec=(256, 32), seed=0)
    assert report.converged
    assert len(report.layers) == 4
    report.assert_healthy(tol=0.2)
