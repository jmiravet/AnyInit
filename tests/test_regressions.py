"""Regression tests for defects found while preparing the paper experiments."""

from __future__ import annotations

import math

import numpy as np
import pytest

import anyinit
from anyinit._run import _VALIDATION_ELEMENTS, _validation_spec

torch = pytest.importorskip("torch")
nn = torch.nn


# --- validation memory --------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(8, 3, 224, 224), (32, 3, 512, 512), (4, 1024), (1, 3, 1024, 1024)]
)
def test_validation_batch_stays_within_its_element_budget(shape):
    widened = _validation_spec(shape)
    assert widened[1:] == shape[1:]
    assert widened[0] >= 1
    if math.prod(shape[1:]) <= _VALIDATION_ELEMENTS:
        assert math.prod(widened) <= _VALIDATION_ELEMENTS


def test_small_inputs_still_get_many_rows():
    assert _validation_spec((32, 64))[0] >= 1024


def test_image_model_validates_with_default_settings():
    """The README's own call: validation must run, not silently vanish."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()
    )
    report = anyinit.initialize(model, "analytic", (32, 3, 224, 224), seed=0)
    assert report.max_deviation is not None
    assert not any("validation skipped" in w for w in report.warnings)


def test_validation_failure_is_reported_not_swallowed():
    def failing_batch():
        raise RuntimeError("CUDA out of memory")

    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
    report = anyinit.initialize(model, input_spec=failing_batch, seed=0)
    assert any("validation skipped" in w and "CUDA out of memory" in w for w in report.warnings)
    assert report.max_deviation is None


# --- registry entries crossing frameworks -----------------------------------------------


@pytest.fixture
def torch_class_activation():
    @anyinit.register_activation(name="t_reg_relu2", overwrite=True)
    class ReLU2(nn.Module):
        def forward(self, x):
            return torch.relu(x) ** 2

    yield ReLU2
    anyinit.unregister_activation("t_reg_relu2")


def test_class_registration_raises_no_unrelated_warning(torch_class_activation):
    model = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
    report = anyinit.initialize(model, input_spec=(32, 8), seed=0)
    assert not any("nested or dynamically built" in w for w in report.warnings)


def test_torch_class_registration_does_not_break_flax(torch_class_activation):
    pytest.importorskip("flax")
    import flax.linen as fnn
    import jax
    import jax.numpy as jnp

    class Net(fnn.Module):
        @fnn.compact
        def __call__(self, x):
            return fnn.Dense(4)(jax.nn.relu(fnn.Dense(8)(x)))

    model = Net()
    params = model.init(jax.random.key(0), jnp.ones((1, 8)))
    _params, report = anyinit.initialize_params(model, params, (16, 8), seed=0)
    assert report.layers[0].activation == "relu"


def test_registered_function_ownership_is_read_from_its_code():
    from anyinit.backends.base import framework_roots

    def uses_torch(x):
        return torch.relu(x) ** 3

    assert "torch" in framework_roots(uses_torch)
    assert "jax" not in framework_roots(uses_torch)


# --- transformer containers and recurrent layers -----------------------------------------


class _TinyTransformer(nn.Module):
    def __init__(self, norm_first: bool, depth: int = 2, dim: int = 32):
        super().__init__()
        self.tok = nn.Embedding(100, dim)
        self.pos = nn.Parameter(torch.zeros(1, 16, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, 4 * dim, batch_first=True, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        return self.head(self.encoder(self.tok(x) + self.pos[:, : x.shape[1]]))


@pytest.mark.parametrize("norm_first", [True, False])
def test_transformer_encoder_weights_are_all_scaled(norm_first):
    model = _TinyTransformer(norm_first)
    report = anyinit.initialize(model, input_spec=torch.randint(0, 100, (8, 16)), seed=0)
    scaled = {record.name for record in report.layers}
    for index in range(2):
        prefix = f"encoder.layers.{index}"
        assert f"{prefix}.linear1" in scaled
        assert f"{prefix}.linear2" in scaled
        assert f"{prefix}.self_attn.in_proj_weight" in scaled
        assert f"{prefix}.self_attn.out_proj.weight" in scaled
    assert [name for name, _ in report.unscaled] == ["pos"]


def test_transformer_layer_activation_is_seen():
    model = _TinyTransformer(norm_first=True)
    report = anyinit.initialize(model, input_spec=torch.randint(0, 100, (8, 16)), seed=0)
    linear1 = next(r for r in report.layers if r.name == "encoder.layers.0.linear1")
    assert linear1.activation.startswith("relu")


def test_transformer_decoder_is_expanded():
    layer = nn.TransformerDecoderLayer(32, 4, 64, batch_first=True)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.TransformerDecoder(layer, 2)

        def forward(self, tgt, memory):
            return self.decoder(tgt, memory)

    from anyinit.backends import resolve

    graph = resolve(Net()).build_graph(Net())
    ops = {n.op for n in graph.scalable}
    assert "MultiheadAttention.in_proj" in ops
    assert sum(1 for n in graph.scalable if n.op == "Linear") == 4


def test_recurrent_weights_are_reported_not_silently_skipped():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(100, 16)
            self.rnn = nn.LSTM(16, 16, batch_first=True)
            self.head = nn.Linear(16, 4)

        def forward(self, x):
            return self.head(self.rnn(self.emb(x))[0][:, -1])

    report = anyinit.initialize(Net(), seed=0)
    names = [name for name, _ in report.unscaled]
    assert names == ["rnn.weight_ih_l0", "rnn.weight_hh_l0"]
    assert all("recurrent" in reason for _, reason in report.unscaled)
    assert "Not scaled" in str(report)


# --- Flax activations reached without going through jax.nn ---------------------------


def _flax_mlp(style: str):
    pytest.importorskip("flax")
    from collections.abc import Callable

    import flax.linen as fnn
    import jax
    import jax.numpy as jnp

    captured = jax.nn.relu

    class Field(fnn.Module):
        act: Callable = jax.nn.relu

        @fnn.compact
        def __call__(self, x):
            for _ in range(3):
                x = self.act(fnn.Dense(64)(x))
            return x

    class Captured(fnn.Module):
        @fnn.compact
        def __call__(self, x):
            for _ in range(3):
                x = captured(fnn.Dense(64)(x))
            return x

    class Lambda(fnn.Module):
        @fnn.compact
        def __call__(self, x):
            for _ in range(3):
                x = (lambda v: jnp.maximum(v, 0.0))(fnn.Dense(64)(x))
            return x

    model = {"field": Field, "captured": Captured, "lambda": Lambda}[style]()
    return model, model.init(jax.random.key(0), jnp.ones((1, 16)))


@pytest.mark.parametrize("style", ["field", "captured", "lambda"])
def test_flax_activation_is_identified_however_it_is_referenced(style):
    model, params = _flax_mlp(style)
    _params, report = anyinit.initialize_params(model, params, (64, 16), seed=0)
    assert [r.activation for r in report.layers] == ["relu", "relu", "relu"]


def test_flax_inferred_activations_are_measured_in_empirical_mode():
    import numpy as np

    model, params = _flax_mlp("field")
    batch = np.random.default_rng(0).standard_normal((512, 16)).astype("float32")
    params, _report = anyinit.initialize_params(model, params, batch, "empirical", seed=0)
    out = np.asarray(model.apply(params, batch))
    assert float((out**2).mean()) == pytest.approx(1.0, rel=0.05)


def test_flax_linear_stack_stays_linear():
    pytest.importorskip("flax")
    import flax.linen as fnn
    import jax
    import jax.numpy as jnp

    class Linear(fnn.Module):
        @fnn.compact
        def __call__(self, x):
            return fnn.Dense(4)(fnn.Dense(32)(x))

    model = Linear()
    params = model.init(jax.random.key(0), jnp.ones((1, 8)))
    _params, report = anyinit.initialize_params(model, params, (32, 8), seed=0)
    assert [r.activation for r in report.layers] == ["none", "none"]
    assert not any("could not identify" in w for w in report.warnings)


def test_flax_matches_pytorch_when_the_activation_is_a_field():
    """The cross-framework guarantee must not depend on how JAX code names its activation."""
    import numpy as np

    model, params = _flax_mlp("field")
    params, _ = anyinit.initialize_params(model, params, (64, 16), seed=3)
    from flax.traverse_util import flatten_dict

    flat = flatten_dict(params["params"])
    flax_kernels = [np.asarray(flat[(f"Dense_{i}", "kernel")]).T for i in range(3)]

    torch_model = nn.Sequential(
        nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
    )
    anyinit.initialize(torch_model, input_spec=(64, 16), seed=3)
    torch_kernels = [torch_model[i].weight.detach().numpy() for i in (0, 2, 4)]
    for a, b in zip(flax_kernels, torch_kernels, strict=True):
        assert np.allclose(a, b, atol=1e-6)


def _pooled_tanh_cnn(nn):
    layers, channels = [], 3
    for _ in range(4):
        layers += [nn.Conv2d(channels, 16, 3, padding=1), nn.Tanh()]
        channels = 16
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10)]
    return nn.Sequential(*layers)


def test_empirical_bounded_activation_before_pooling_settles():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    batch = torch.randn(32, 3, 16, 16)
    stds = {}
    for mode in ("analytic", "empirical"):
        torch.manual_seed(0)
        model = _pooled_tanh_cnn(nn)
        result = anyinit.initialize(model, mode, batch, seed=0)
        assert not any("did not settle" in w or "out of reach" in w for w in result.warnings)
        stds[mode] = [float(m.weight.std()) for m in model if hasattr(m, "weight")]
    for analytic, empirical in zip(stds["analytic"], stds["empirical"], strict=True):
        assert math.isfinite(empirical)
        assert empirical == pytest.approx(analytic, rel=0.1)


def test_empirical_unreachable_objective_is_bounded_and_reported():
    from anyinit.core.empirical import MAX_DRIFT, _solve_layer
    from anyinit.core.moments import MomentState
    from anyinit.core.solve import Objective, SolveResult

    applied = []
    result = SolveResult()

    # The measured second moment saturates at 0.1 whatever the scale.
    def measure(points):
        m2 = min(0.1, 0.01 * result.scales.get("w", 1.0) ** 2)
        return {points[0]: MomentState(0.0, m2, 3 * m2 * m2)}

    _solve_layer(
        "w",
        "a",
        Objective("m2", 1.0),
        None,
        result,
        measure,
        lambda _, s: applied.append(s),
        max_iterations=50,
    )
    assert result.scales["w"] <= MAX_DRIFT
    assert any("out of reach" in w or "did not settle" in w for w in result.warnings)


@pytest.mark.parametrize("mode", ["analytic", "empirical"])
def test_keras_sequential_with_input_is_measured(mode):
    keras = pytest.importorskip("keras")
    keras.utils.set_random_seed(0)
    layers = [keras.Input((8, 8, 3))]
    for _ in range(3):
        layers += [keras.layers.Conv2D(8, 3, padding="same"), keras.layers.Activation("tanh")]
    layers += [keras.layers.GlobalAveragePooling2D(), keras.layers.Dense(4)]
    model = keras.Sequential(layers)
    batch = np.random.default_rng(0).standard_normal((16, 8, 8, 3)).astype("float32")
    report = anyinit.initialize(model, mode, batch, seed=0)
    assert not any("no measurement" in w for w in report.warnings)
    tanh_rows = [layer for layer in report.layers if layer.activation == "tanh"]
    assert len(tanh_rows) == 3
    for row in tanh_rows:
        assert row.measured is not None
        assert row.measured == pytest.approx(0.5, abs=0.08)


# --- identity layers ----------------------------------------------------------------------


def test_torch_identity_is_a_pass_through_not_an_activation():
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Identity(), nn.Linear(8, 8), nn.ReLU())
    with_identity = anyinit.initialize(model, input_spec=(4, 8), seed=0)
    without = anyinit.initialize(
        nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU()),
        input_spec=(4, 8),
        seed=0,
    )
    assert [r.activation for r in with_identity.layers] == ["relu", "relu"]
    assert [r.scale for r in with_identity.layers] == [r.scale for r in without.layers]


@pytest.mark.parametrize("make", ["identity", "linear"])
def test_keras_identity_is_a_pass_through_not_an_activation(make):
    keras = pytest.importorskip("keras")
    passthrough = (
        keras.layers.Identity() if make == "identity" else keras.layers.Activation("linear")
    )
    model = keras.Sequential(
        [keras.Input((8,)), keras.layers.Dense(8), passthrough, keras.layers.Dense(8, "relu")]
    )
    report = anyinit.initialize(model, input_spec=(4, 8), seed=0)
    assert "identity" not in [r.activation for r in report.layers]


# --- empirical mode inside expanded containers, and with dropout ---------------------------


def test_empirical_measures_inside_transformer_layers():
    torch.manual_seed(0)
    model = _TinyTransformer(norm_first=False)
    report = anyinit.initialize(model, "empirical", torch.randint(0, 100, (8, 16)), seed=0)
    assert not any("settle" in w or "no measurement" in w for w in report.warnings)
    feeds_relu = [r for r in report.layers if r.name.endswith("linear1")]
    assert len(feeds_relu) == 2
    for row in feeds_relu:
        assert row.measured == pytest.approx(1.0, abs=0.01)


def test_measurements_leave_the_callers_random_stream_alone():
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 4))
    batch = torch.randn(64, 16)
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    anyinit.initialize(model, "empirical", batch, seed=0)
    assert torch.equal(torch.rand(3), expected)


def test_keras_empirical_with_dropout_settles():
    keras = pytest.importorskip("keras")
    keras.utils.set_random_seed(0)
    layers = [keras.Input((16,))]
    for _ in range(3):
        layers += [keras.layers.Dense(32, "relu"), keras.layers.Dropout(0.3)]
    model = keras.Sequential([*layers, keras.layers.Dense(4)])
    batch = np.random.default_rng(0).standard_normal((64, 16)).astype("float32")
    report = anyinit.initialize(model, "empirical", batch, seed=0)
    assert not any("settle" in w for w in report.warnings)


class _Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(x + self.bn2(self.conv2(h)))


def test_empirical_residual_floor_is_reported_not_spun_on():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), _Block(16), _Block(16)
    )
    report = anyinit.initialize(model, "empirical", torch.randn(16, 3, 16, 16), seed=0)
    assert not any("settle" in w for w in report.warnings)
    for row in report.layers:
        if row.name.endswith("bn2"):
            assert row.measured == pytest.approx(1.0, abs=0.02)


# --- adaptive pooling ---------------------------------------------------------------------


def test_adaptive_pool_window_comes_from_the_input_shape():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )
    report = anyinit.initialize(model, "analytic", (64, 3, 16, 16), seed=0)
    pooled = next(r for r in report.layers if "AdaptiveAvgPool2d" in (r.activation or ""))
    assert pooled.measured is not None
    assert pooled.predicted is not None
    # Treated as a pass-through, the 16x16 average was predicted at 1 and measured near 0.4.
    assert pooled.measured == pytest.approx(pooled.predicted, rel=0.25)
