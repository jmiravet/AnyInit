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
