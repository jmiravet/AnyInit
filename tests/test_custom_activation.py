"""User-defined activations, which is what the library is for.

``relu(x) ** 3`` exercises every part at once: it is not in any table, it has a kink,
its moments have closed forms to check against, and it is homogeneous of degree three --
which makes it provably impossible to stabilize across depth, so the library has to say
so rather than hand back a dead network.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import anyinit
from anyinit.core.graph import NodeKind

torch = pytest.importorskip("torch")
nn = torch.nn

# E[relu(x)^3] = sqrt(2/pi);  E[relu(x)^6] = 15/2
EXACT_MEAN = math.sqrt(2.0 / math.pi)
EXACT_M2 = 7.5
EXACT_SIGMA_STAR = (1.0 / 7.5) ** (1.0 / 6.0)


@pytest.fixture
def relu3_numpy():
    anyinit.register_activation(
        lambda x: np.maximum(x, 0.0) ** 3, name="t_relu3_np", overwrite=True
    )
    yield "t_relu3_np"
    anyinit.unregister_activation("t_relu3_np")


def module_level_relu3(x):
    """Defined at module level on purpose: see ``test_nested_function_is_reported``."""
    return torch.relu(x) ** 3


@pytest.fixture
def relu3_function():
    anyinit.register_activation(module_level_relu3, name="t_relu3_fn", overwrite=True)
    yield "t_relu3_fn", module_level_relu3
    anyinit.unregister_activation("t_relu3_fn")


@pytest.fixture
def relu3_module():
    @anyinit.register_activation(name="t_relu3_mod", overwrite=True)
    class ReLU3(nn.Module):
        def forward(self, x):
            return torch.relu(x) ** 3

    yield "t_relu3_mod", ReLU3
    anyinit.unregister_activation("t_relu3_mod")


def _backend():
    from anyinit.backends import resolve

    return resolve(nn.Linear(2, 2))


def test_numpy_registration_needs_no_backend(relu3_numpy):
    profile = anyinit.activation_profile(relu3_numpy)
    assert profile.moments(0.0, 1.0).m2 == pytest.approx(EXACT_M2, abs=1e-10)


@pytest.mark.parametrize("fixture_name", ["relu3_numpy", "relu3_function", "relu3_module"])
def test_every_registration_form_gives_the_same_exact_profile(fixture_name, request):
    value = request.getfixturevalue(fixture_name)
    name = value if isinstance(value, str) else value[0]
    profile = anyinit.activation_profile(name, _backend())
    state = profile.moments(0.0, 1.0)

    assert state.mean == pytest.approx(EXACT_MEAN, abs=1e-9)
    assert state.m2 == pytest.approx(EXACT_M2, abs=1e-8)
    assert profile.homogeneous_degree == pytest.approx(3.0)
    assert profile.chi == pytest.approx(3.0, abs=1e-6)
    assert profile.diagnostics.sigma_star == pytest.approx(EXACT_SIGMA_STAR, abs=1e-6)


def test_registered_module_is_one_graph_node_not_two(relu3_module):
    """Check that a registered module stays a single graph node.

    Without registration FX inlines it into ``relu`` and ``pow``, losing the fact that
    it is one activation with one profile.
    """
    _name, cls = relu3_module

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 64)
            self.act = cls()
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    graph = _backend().build_graph(Net())
    activations = graph.of_kind(NodeKind.ACTIVATION)
    assert [n.op for n in activations] == ["t_relu3_mod"]


class FunctionNet(nn.Module):
    """Calls the activation by its module-level name, as a user would write it.

    The name matters: FX's wrap registry patches a module dict, so the call site has to
    resolve through that dict.  A closure variable bypasses it.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(module_level_relu3(self.fc1(x)))


def test_registered_function_is_one_graph_node(relu3_function):
    """FX keeps a module-level registered function atomic via its wrap registry."""
    graph = _backend().build_graph(FunctionNet())
    assert [n.op for n in graph.of_kind(NodeKind.ACTIVATION)] == ["t_relu3_fn"]


@pytest.mark.parametrize("mode", ["analytic", "empirical"])
def test_first_layer_hits_the_target(relu3_module, mode):
    """Depth one is solvable even for an unstable activation, and must be solved."""
    _name, cls = relu3_module
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(128, 128, bias=False), cls())
    anyinit.initialize(model, mode, (2048, 128), seed=0)

    with torch.no_grad():
        output = model(torch.randn(8192, 128))
    assert float(output.pow(2).mean()) == pytest.approx(1.0, rel=0.35)


def test_unstable_activation_is_reported_not_silently_accepted(relu3_module):
    _name, cls = relu3_module
    torch.manual_seed(0)
    layers = []
    for _ in range(20):
        layers += [nn.Linear(128, 128, bias=False), cls()]
    report = anyinit.initialize(nn.Sequential(*layers), input_spec=(512, 128), seed=0)

    record = next(s for s in report.stability if s.activation.startswith("t_relu3"))
    assert record.verdict == "unstable"
    assert record.chi == pytest.approx(3.0, abs=1e-6)
    assert record.drift == pytest.approx(3.0**20, rel=1e-6)
    assert "no scalar initialization is depth-stable" in record.advice().lower()

    with pytest.raises(AssertionError, match="not depth-stable"):
        report.assert_healthy()


def test_assert_healthy_refuses_an_unstable_network(relu3_module):
    _name, cls = relu3_module
    torch.manual_seed(0)
    layers = []
    for _ in range(6):
        layers += [nn.Linear(64, 64, bias=False), cls()]
    report = anyinit.initialize(nn.Sequential(*layers), input_spec=(256, 64), seed=0)
    with pytest.raises(AssertionError, match="not depth-stable"):
        report.assert_healthy()


def test_relu_squared_is_degree_two():
    """The PINN case: relu(x)**2 at shallow depth, which is where it is usable."""
    anyinit.register_activation(lambda x: np.maximum(x, 0.0) ** 2, name="t_relu2", overwrite=True)
    try:
        profile = anyinit.activation_profile("t_relu2")
        assert profile.homogeneous_degree == pytest.approx(2.0)
        assert profile.moments(0.0, 1.0).m2 == pytest.approx(1.5, abs=1e-10)
        assert profile.diagnostics.sigma_star == pytest.approx((1.0 / 1.5) ** 0.25, abs=1e-6)
    finally:
        anyinit.unregister_activation("t_relu2")


def test_nested_function_cannot_be_kept_atomic_and_says_so():
    """Check that a nested function's limitation is reported.

    FX's wrap mechanism swaps a module-level name, so a closure cannot be reached, and
    the report has to say so rather than scaling for the wrong activation.
    """

    def nested(x):
        return torch.relu(x) ** 3

    anyinit.register_activation(nested, name="t_relu3_nested", overwrite=True)
    try:
        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
        report = anyinit.initialize(model, input_spec=(64, 16), seed=0)
        assert any("cannot keep it atomic" in w for w in report.warnings)
    finally:
        anyinit.unregister_activation("t_relu3_nested")
