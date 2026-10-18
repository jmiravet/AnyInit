"""PyTorch specifics: layouts, layer families, and the FX fallback."""

from __future__ import annotations

import numpy as np
import pytest

import anyinit
from anyinit.core.graph import FIDELITY_GRAPH, FIDELITY_LINEAR, NodeKind

torch = pytest.importorskip("torch")
nn = torch.nn


@pytest.fixture
def backend():
    from anyinit.backends import resolve

    return resolve(nn.Linear(2, 2))


def test_detects_torch_modules(backend):
    assert backend.name == "pytorch"


def test_embedding_is_not_scaled_by_fan():
    """Check that an embedding keeps its own scale.

    A lookup returns a row of the table, so there is no sum over fan_in to compensate
    for, and compensating anyway would shrink the variance by the embedding width.
    """
    embedding_dim = 512

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(1000, embedding_dim)
            self.fc = nn.Linear(embedding_dim, 10)

        def forward(self, idx):
            return self.fc(torch.relu(self.emb(idx)))

    model = Net()
    anyinit.initialize(model, seed=0)
    std = float(model.emb.weight.detach().std())

    # Scaled for the ReLU that follows it, so sqrt(2); emphatically not 1/sqrt(512).
    assert std == pytest.approx(np.sqrt(2.0), rel=0.05)
    assert std > 10.0 / np.sqrt(embedding_dim)


def test_conv_transpose_fan_is_not_inverted(backend):
    """Check the fan of a transposed convolution.

    Torch stores its kernel as (in, out/groups, *k), so a naive reading of the native
    shape gives 288 where the operation's fan_in is 144.
    """
    graph = backend.build_graph(_wrap(nn.ConvTranspose2d(16, 32, 3)))
    spec = next(n.spec for n in graph.of_kind(NodeKind.PARAMETRIC))
    from anyinit.core.fan import fan_in

    assert spec.canonical_shape == (32, 16, 3, 3)
    assert fan_in(spec) == 144


@pytest.mark.parametrize(
    "layer",
    [
        nn.Linear(16, 8),
        nn.Conv1d(4, 8, 3),
        nn.Conv2d(4, 8, 3),
        nn.Conv3d(4, 8, 3),
        nn.ConvTranspose1d(4, 8, 3),
        nn.ConvTranspose2d(4, 8, 3),
        nn.Conv2d(8, 8, 3, groups=8),
        nn.Conv2d(8, 16, 3, groups=4),
    ],
    ids=lambda layer: (
        type(layer).__name__
        + (f"_g{layer.groups}" if hasattr(layer, "groups") and layer.groups > 1 else "")
    ),
)
def test_weight_layout_round_trips(backend, layer):
    """Canonical -> native -> canonical must be the identity for every layer family."""
    graph = backend.build_graph(_wrap(layer))
    node = next(n for n in graph.of_kind(NodeKind.PARAMETRIC))
    original = np.arange(node.spec.size, dtype=np.float64).reshape(node.spec.canonical_shape)
    backend.write_weight(node.handle, original)
    assert np.allclose(backend.read_weight(node.handle), original)


def test_one_by_one_weight_does_not_become_nan():
    """Check that a single-element weight stays finite.

    It has no unbiased variance, so a naive normalization produces NaN.
    """
    model = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1))
    anyinit.initialize(model, input_spec=(8, 1), seed=0)
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()


def test_norm_without_affine_is_handled(backend):
    model = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8, affine=False), nn.ReLU())
    report = anyinit.initialize(model, input_spec=(32, 8), seed=0)
    assert report.converged


def test_attention_splits_into_two_blocks(backend):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(64, 4, batch_first=True)

        def forward(self, x):
            return self.attn(x, x, x)[0]

    graph = backend.build_graph(Net())
    parametric = graph.of_kind(NodeKind.PARAMETRIC)
    assert [n.op for n in parametric] == [
        "MultiheadAttention.in_proj",
        "MultiheadAttention.out_proj",
    ]
    # Packed q/k/v is three (E, E) maps, so fan_in is E and not 3E.
    from anyinit.core.fan import fan_in

    assert fan_in(parametric[0].spec) == 64


def test_residual_block_gives_both_branches_the_same_activation(backend):
    """The ResNet transition block that v1 mis-attributed."""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 8, 3, padding=1)
            self.bn = nn.BatchNorm2d(8)
            self.down = nn.Conv2d(4, 8, 1)
            self.down_bn = nn.BatchNorm2d(8)

        def forward(self, x):
            return torch.relu(self.bn(self.conv(x)) + self.down_bn(self.down(x)))

    from anyinit.core.topology import build_plan

    graph = backend.build_graph(Block())
    assert graph.fidelity == FIDELITY_GRAPH
    plan = build_plan(graph)
    relu = next(n.id for n in graph.of_kind(NodeKind.ACTIVATION))
    ancestors = plan.ancestors[relu]
    assert len(ancestors) == 2, f"expected both normalizations, got {ancestors}"


def test_control_flow_falls_back_to_a_linear_graph(backend):
    class Dynamic(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)

        def forward(self, x):
            if float(x.sum()) > 0:
                x = x * 2
            return self.fc(x)

    graph = backend.build_graph(Dynamic())
    assert graph.fidelity == FIDELITY_LINEAR
    assert any("could not trace" in note for note in graph.notes)


class _ActivationFirst(nn.Module):
    """Registers its activation before the layer feeding it, as Hugging Face's CLIP does."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.act = nn.GELU()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.bn = nn.BatchNorm1d(64)

    def forward(self, x):
        if float(x.sum()) > 1e9:  # Data-dependent control flow: FX cannot trace this.
            x = x * 2
        return self.bn(self.fc2(self.act(self.fc1(self.norm(x)))))


def test_fallback_follows_execution_order(backend):
    model = _ActivationFirst()
    graph = backend.build_graph(model, (16, 64))
    assert graph.fidelity == FIDELITY_LINEAR
    assert list(graph.order[1:-1]) == ["norm", "fc1", "act", "fc2", "bn"]
    assert any("execution order" in note for note in graph.notes)

    report = anyinit.initialize(model, "analytic", (16, 64), seed=0)
    by_name = {layer.name: layer.activation for layer in report.layers}
    assert by_name["fc1"] == "gelu"
    assert by_name["norm"] == "none"


def test_fallback_probe_leaves_the_model_as_it_was(backend):
    model = _ActivationFirst().train()
    before = model.bn.running_mean.clone(), int(model.bn.num_batches_tracked)
    backend.build_graph(model, (16, 64))
    assert model.training
    assert torch.equal(model.bn.running_mean, before[0])
    assert int(model.bn.num_batches_tracked) == before[1]
    assert not any(m._forward_pre_hooks for m in model.modules())


def test_fallback_without_a_probe_uses_registration_order(backend):
    graph = backend.build_graph(_ActivationFirst())
    assert list(graph.order[1:-1]) == ["norm", "act", "fc1", "fc2", "bn"]
    assert any("registration order" in note and "no input shape" in note for note in graph.notes)


def test_fallback_when_the_probe_raises_uses_registration_order(backend):
    class Broken(_ActivationFirst):
        def forward(self, x):
            if float(x.sum()) == 0:  # Untraceable, and fails on the all-zeros probe.
                raise ValueError("probe")
            return super().forward(x)

    graph = backend.build_graph(Broken(), (16, 64))
    assert list(graph.order[1:-1]) == ["norm", "act", "fc1", "fc2", "bn"]
    assert any("raised ValueError" in note for note in graph.notes)


def test_transformer_encoder_layer_does_not_raise():
    """A stock PyTorch module that symbolic tracing cannot handle."""
    layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
    report = anyinit.initialize(layer, seed=0)
    assert report.graph_fidelity == FIDELITY_LINEAR
    assert report.layers


def test_leaky_relu_slope_reaches_the_profile(backend):
    gentle = nn.Sequential(nn.Linear(64, 64), nn.LeakyReLU(0.01))
    steep = nn.Sequential(nn.Linear(64, 64), nn.LeakyReLU(0.5))
    a = anyinit.initialize(gentle, input_spec=(64, 64), seed=0)
    b = anyinit.initialize(steep, input_spec=(64, 64), seed=0)
    assert a.layers[0].scale != pytest.approx(b.layers[0].scale)


def test_dropout_is_accounted_for(backend):
    """Inverted dropout multiplies the variance reaching the next layer by 1/(1-p)."""
    with_dropout = nn.Sequential(nn.Linear(64, 64), nn.Dropout(0.5), nn.ReLU(), nn.Linear(64, 8))
    without = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))
    a = anyinit.initialize(with_dropout, input_spec=(64, 64), seed=0)
    b = anyinit.initialize(without, input_spec=(64, 64), seed=0)
    assert a.layers[0].scale == pytest.approx(b.layers[0].scale / np.sqrt(2.0), rel=1e-6)


def _wrap(layer):
    """Put a bare layer in a traceable module."""

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = layer

        def forward(self, x):
            return self.inner(x)

    return Wrapper()


def test_deep_relu_stack_is_healthy():
    """One draw of a deep stack wanders from the ensemble prediction; that is not a fault."""
    model = nn.Sequential(*[m for _ in range(30) for m in (nn.Linear(256, 256), nn.ReLU())])
    report = anyinit.initialize(model, input_spec=(64, 256), seed=0)
    report.assert_healthy()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_model_on_cuda_is_validated():
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10)).cuda()
    report = anyinit.initialize(model, input_spec=(32, 64), seed=0)
    assert not report.warnings, report.warnings
    assert all(r.measured is not None for r in report.layers)


def test_validation_batch_follows_the_model_dtype():
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU()).double()
    report = anyinit.initialize(model, input_spec=(32, 16), seed=0)
    assert not report.warnings, report.warnings
    assert report.layers[0].measured is not None


def test_initializing_does_not_load_dynamo():
    """torch._dynamo pulls in Triton, which crashes a process that loaded TensorFlow first."""
    import subprocess
    import sys

    code = (
        "import sys, torch, anyinit; from torch import nn; "
        "m = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(2)); "
        "anyinit.initialize(m, input_spec=(2, 3, 8, 8)); "
        "assert 'torch._dynamo' not in sys.modules"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


class _TiedLM(nn.Module):
    """An embedding whose table is also the output layer, written one of four ways."""

    def __init__(self, head: str, vocab: int = 500, width: int = 64):
        super().__init__()
        self.form = head
        self.emb = nn.Embedding(vocab, width)
        self.fc = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        if head == "module":
            self.head = nn.Linear(width, vocab, bias=False)
            self.head.weight = self.emb.weight

    def forward(self, idx):
        h = self.norm(torch.relu(self.fc(self.emb(idx))))
        if self.form == "module":
            return self.head(h)
        if self.form == "linear":
            return torch.nn.functional.linear(h, self.emb.weight)
        if self.form == "T":
            return h @ self.emb.weight.T
        return torch.matmul(h, self.emb.weight.t())


@pytest.mark.parametrize("head", ["module", "linear", "T", "t"])
@pytest.mark.parametrize("mode", ["analytic", "empirical"])
def test_tied_table_takes_the_output_scale(head, mode):
    from anyinit.core.tying import DOCS

    model = _TiedLM(head)
    idx = torch.randint(0, 500, (16, 32), generator=torch.Generator().manual_seed(0))
    report = anyinit.initialize(model, mode, input_spec=idx, seed=0)

    assert float(model.emb.weight.detach().std()) == pytest.approx(64**-0.5, rel=1e-2)
    with torch.no_grad():
        assert float(model(idx).var()) == pytest.approx(1.0, rel=0.1)
    assert any(DOCS in w for w in report.warnings)


def test_untied_embedding_keeps_the_lookup_scale():
    model = _TiedLM("module")
    model.head.weight = nn.Parameter(torch.empty(500, 64))
    report = anyinit.initialize(model, seed=0)

    assert float(model.emb.weight.detach().std()) == pytest.approx(1.0, rel=1e-2)
    assert not any("tied embedding" in w for w in report.warnings)


class _Scaled(nn.Module):
    """``Linear -> times 4 -> ReLU -> Linear``, the factor written one of several ways."""

    def __init__(self, form: str):
        super().__init__()
        self.form = form
        self.fc, self.out = nn.Linear(64, 64), nn.Linear(64, 10)
        self.register_buffer("four", torch.tensor(4.0))

    def forward(self, x):
        h = self.fc(x)
        h = {
            "x * c": lambda: h * 4.0,
            "c * x": lambda: 4.0 * h,
            "x / c": lambda: h / 0.25,
            "x.mul(c)": lambda: h.mul(4.0),
            "buffer": lambda: h * self.four,
        }[self.form]()
        return self.out(torch.relu(h))


@pytest.mark.parametrize("form", ["x * c", "c * x", "x / c", "x.mul(c)", "buffer"])
def test_a_constant_factor_is_solved_around(form):
    model = _Scaled(form)
    report = anyinit.initialize(model, input_spec=(1024, 64), seed=0)

    assert report.layers[0].scale == pytest.approx((2.0 / 64) ** 0.5 / 4.0)
    report.assert_healthy()


def test_a_per_channel_constant_is_solved_around():
    class LayerScale(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc, self.out = nn.Linear(64, 64), nn.Linear(64, 10)
            self.gamma = nn.Parameter(torch.linspace(1.0, 3.0, 64))

        def forward(self, x):
            return self.out(torch.relu(self.fc(x) * self.gamma))

    model = LayerScale()
    report = anyinit.initialize(model, input_spec=(1024, 64), seed=0)
    report.assert_healthy()


def test_a_lookup_multiplier_reaches_the_layer_reading_it():
    """``emb(idx) * sqrt(d)`` feeding a layer with no normalization in between."""

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.fc = nn.Embedding(500, 64), nn.Linear(64, 64)

        def forward(self, idx):
            return torch.relu(self.fc(self.emb(idx) * 8.0))

    model = Net()
    idx = torch.randint(0, 500, (64, 32), generator=torch.Generator().manual_seed(0))
    anyinit.initialize(model, input_spec=idx, seed=0)
    with torch.no_grad():
        assert float(model(idx).pow(2).mean()) == pytest.approx(1.0, rel=0.1)


def test_a_logit_multiplier_gives_the_tied_table_a_scale_of_one():
    """PaLM: logits times 1/sqrt(d) let one scale serve both roles."""

    class PaLMHead(_TiedLM):
        def forward(self, idx):
            return super().forward(idx) * 64**-0.5

    model = PaLMHead("module")
    idx = torch.randint(0, 500, (16, 32), generator=torch.Generator().manual_seed(0))
    anyinit.initialize(model, input_spec=idx, seed=0)

    assert float(model.emb.weight.detach().std()) == pytest.approx(1.0, rel=1e-2)
    with torch.no_grad():
        assert float(model(idx).var()) == pytest.approx(1.0, rel=0.1)


# ------------------------------------------------------------- activations


class _Fn(nn.Module):
    """A functional activation called inside ``forward()``."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def _activation_of(backend, act):
    graph = backend.build_graph(nn.Sequential(nn.Linear(8, 8), act, nn.Linear(8, 8)))
    (node,) = graph.of_kind(NodeKind.ACTIVATION)
    return str(node.meta["activation"])


F = torch.nn.functional


@pytest.mark.parametrize(
    ("act", "expected"),
    [
        (nn.PReLU(), "leaky_relu(negative_slope=0.25)"),
        (nn.PReLU(8, init=0.1), "leaky_relu(negative_slope=0.1)"),
        (nn.GELU(approximate="tanh"), "gelu_tanh"),
        (nn.LogSigmoid(), "log_sigmoid"),
        (nn.Hardshrink(0.3), "hardshrink(lambd=0.3)"),
        (nn.Softshrink(), "softshrink(lambd=0.5)"),
        (nn.Tanhshrink(), "tanhshrink"),
        (nn.Threshold(0.1, -0.5), "threshold(threshold=0.1, value=-0.5)"),
        (nn.Softmin(dim=-1), "softmin"),
        (_Fn(lambda x: F.gelu(x, approximate="tanh")), "gelu_tanh"),
        (_Fn(lambda x: F.hardtanh(x, 0.0, 6.0)), "hardtanh(hi=6, lo=0)"),
        (_Fn(lambda x: F.rrelu(x, 0.1, 0.3)), "leaky_relu(negative_slope=0.2)"),
        (_Fn(F.logsigmoid), "log_sigmoid"),
        (_Fn(lambda x: F.softshrink(x, 0.2)), "softshrink(lambd=0.2)"),
        (_Fn(lambda x: F.threshold(x, 0.1, 0.0)), "threshold(threshold=0.1, value=0)"),
        (_Fn(torch.relu_), "relu"),
        (_Fn(lambda x: x.sigmoid_()), "sigmoid"),
    ],
    ids=lambda v: v if isinstance(v, str) else type(v).__name__,
)
def test_activation_is_recognized_with_its_parameters(backend, act, expected):
    assert _activation_of(backend, act) == expected


def test_functional_prelu_reads_its_weight_and_leaves_it_out_of_the_signal(backend):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)
            self.slope = nn.Parameter(torch.full((8,), 0.2))

        def forward(self, x):
            return F.prelu(self.fc(x), self.slope)

    graph = backend.build_graph(Net())
    (node,) = graph.of_kind(NodeKind.ACTIVATION)
    assert str(node.meta["activation"]) == "leaky_relu(negative_slope=0.2)"
    assert [graph[i].kind for i in node.inputs] == [NodeKind.PARAMETRIC]


@pytest.mark.parametrize(
    ("name", "kwargs", "native"),
    [
        ("gelu_tanh", {}, lambda x: F.gelu(x, approximate="tanh")),
        ("quick_gelu", {}, lambda x: x * torch.sigmoid(1.702 * x)),
        ("relu2", {}, lambda x: torch.relu(x) ** 2),
        ("log_sigmoid", {}, F.logsigmoid),
        ("sqrt_softplus", {}, lambda x: F.softplus(x).sqrt()),
        ("clipped_gelu", {"lo": -1.0, "hi": 1.0}, lambda x: F.gelu(x).clip(-1.0, 1.0)),
        ("hardshrink", {"lambd": 0.3}, lambda x: F.hardshrink(x, 0.3)),
        ("softshrink", {"lambd": 0.3}, lambda x: F.softshrink(x, 0.3)),
        ("tanhshrink", {}, F.tanhshrink),
        ("threshold", {"threshold": 0.1, "value": -0.5}, lambda x: F.threshold(x, 0.1, -0.5)),
        ("hard_mish", {}, lambda x: 0.5 * x * (x + 2).clamp(0, 2)),
        (
            "laplace",
            {},
            lambda x: 0.5 * (1 + torch.erf((x - 0.707107) / (0.282095 * 2**0.5))),
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_numpy_reference_matches_torch(name, kwargs, native):
    from anyinit.core.activations import BUILTIN

    x = np.linspace(-12.0, 12.0, 4001)
    expected = native(torch.from_numpy(x)).numpy()
    assert np.allclose(BUILTIN[name](x, **kwargs), expected, rtol=0, atol=1e-12)


def _hf(name):
    pytest.importorskip("transformers")
    from transformers.activations import ACT2FN

    return ACT2FN[name]


def _timm(name):
    pytest.importorskip("timm")
    from timm.layers import activations, activations_me

    return getattr(activations, name, None) or getattr(activations_me, name)


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (lambda: _hf("gelu_new"), "gelu_tanh"),
        (lambda: _hf("gelu_fast"), "gelu_tanh"),
        (lambda: _hf("gelu_accurate"), "gelu_tanh"),
        (lambda: _hf("gelu_python_tanh"), "gelu_tanh"),
        (lambda: _hf("gelu_python"), "gelu"),
        (lambda: _hf("gelu_10"), "clipped_gelu(hi=10, lo=-10)"),
        (lambda: _hf("quick_gelu"), "quick_gelu"),
        (lambda: _hf("relu2"), "relu2"),
        (lambda: _hf("sqrtsoftplus"), "sqrt_softplus"),
        (lambda: _hf("laplace"), "laplace"),
        (lambda: _timm("Swish")(), "silu"),
        (lambda: _timm("HardSwish")(), "hardswish"),
        (lambda: _timm("HardMish")(), "hard_mish"),
        (lambda: _timm("QuickGELU")(), "quick_gelu"),
        (lambda: _timm("PReLU")(), "leaky_relu(negative_slope=0.25)"),
        (lambda: _timm("HardSigmoidMe")(), "hardsigmoid"),
    ],
    ids=[
        "hf-gelu_new", "hf-gelu_fast", "hf-gelu_accurate", "hf-gelu_python_tanh",
        "hf-gelu_python", "hf-gelu_10", "hf-quick_gelu", "hf-relu2", "hf-sqrtsoftplus",
        "hf-laplace", "timm-Swish", "timm-HardSwish", "timm-HardMish", "timm-QuickGELU",
        "timm-PReLU", "timm-HardSigmoidMe",
    ],
)  # fmt: skip
def test_third_party_activation_class_is_kept_whole(backend, make, expected):
    """Check that a library's activation class is one node, not the primitives inside it."""
    assert _activation_of(backend, make()) == expected


def test_third_party_activation_is_scaled_for(backend):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(256, 256), _hf("gelu_new"), nn.Linear(256, 256))
    report = anyinit.initialize(model, "analytic", (2048, 256), seed=0)
    assert report.layers[0].activation == "gelu_tanh"
    assert report.layers[-1].measured == pytest.approx(1.0, rel=0.05)


class QuickGELUActivation(nn.Module):
    """Shares a transformers class name, but is not that class."""

    def forward(self, x):
        return torch.tanh(x)


def test_class_sharing_a_third_party_name_is_traced_into(backend):
    assert _activation_of(backend, QuickGELUActivation()) == "tanh"
