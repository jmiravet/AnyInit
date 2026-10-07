"""Pairing layers with activations by walking edges, not list order."""

from __future__ import annotations

from anyinit.core.fan import ParamSpec
from anyinit.core.graph import ModelGraph, Node, NodeKind
from anyinit.core.registry import ActivationRef
from anyinit.core.topology import (
    build_plan,
    detect_shared,
    dominating_scalable_ancestors,
    measurement_node,
    repair_path,
)


def test_straight_chain_pairs_each_layer_with_the_next_activation(mlp_graph):
    graph = mlp_graph(depth=3)
    plan = build_plan(graph)
    assert plan.measurement == {"fc0": "act0", "fc1": "act1", "fc2": "act2"}
    assert plan.ancestors == {"act0": ("fc0",), "act1": ("fc1",), "act2": ("fc2",)}


def test_both_branches_of_a_merge_are_dominating_ancestors(residual_graph):
    """The regression this module exists for.

    A list-slicing scheme puts bn_main and bn_skip in different blocks and gives the
    post-addition ReLU's gain to whichever is listed second.  Both feed it.
    """
    ancestors = dominating_scalable_ancestors(residual_graph, "relu")
    assert set(ancestors) == {"bn_main", "bn_skip"}


def test_both_branches_measure_at_the_same_activation(residual_graph):
    plan = build_plan(residual_graph)
    assert plan.measurement["bn_main"] == "relu"
    assert plan.measurement["bn_skip"] == "relu"


def test_search_walks_through_transparent_nodes():
    nodes = [
        Node("x", NodeKind.INPUT, "input"),
        Node("conv", NodeKind.PARAMETRIC, "Conv2d", ("x",), spec=ParamSpec("conv", (8, 4, 3, 3))),
        Node("pool", NodeKind.POOL, "MaxPool2d", ("conv",), meta={"pool": "max", "window": 4}),
        Node("drop", NodeKind.DROPOUT, "Dropout", ("pool",), meta={"p": 0.5}),
        Node("flat", NodeKind.SHAPE, "Flatten", ("drop",)),
        Node(
            "relu",
            NodeKind.ACTIVATION,
            "relu",
            ("flat",),
            meta={"activation": ActivationRef("relu")},
        ),
        Node("out", NodeKind.OUTPUT, "output", ("relu",)),
    ]
    graph = ModelGraph(nodes)
    assert measurement_node(graph, "conv") == "relu"
    assert dominating_scalable_ancestors(graph, "relu") == ("conv",)


def test_search_stops_at_an_intervening_activation(mlp_graph):
    graph = mlp_graph(depth=3)
    # act1 belongs to fc1, not fc0: the walk back from act1 must stop at act0.
    assert dominating_scalable_ancestors(graph, "act1") == ("fc1",)


def test_layer_with_nothing_after_it_measures_itself():
    nodes = [
        Node("x", NodeKind.INPUT, "input"),
        Node("fc", NodeKind.PARAMETRIC, "Linear", ("x",), spec=ParamSpec("dense", (4, 4))),
        Node("out", NodeKind.OUTPUT, "output", ("fc",)),
    ]
    assert measurement_node(ModelGraph(nodes), "fc") == "fc"


def test_repair_path_covers_both_branches(residual_graph):
    path = repair_path(residual_graph, "relu", ("bn_main", "bn_skip"))
    assert set(path) == {"bn_main", "bn_skip", "add", "relu"}
    order = {nid: i for i, nid in enumerate(residual_graph.order)}
    assert list(path) == sorted(path, key=lambda n: order[n])


def test_shared_layers_are_detected():
    nodes = [
        Node("x", NodeKind.INPUT, "input"),
        Node(
            "a",
            NodeKind.PARAMETRIC,
            "Linear",
            ("x",),
            spec=ParamSpec("dense", (4, 4)),
            meta={"shared_key": "m1"},
        ),
        Node(
            "relu", NodeKind.ACTIVATION, "relu", ("a",), meta={"activation": ActivationRef("relu")}
        ),
        Node(
            "b",
            NodeKind.PARAMETRIC,
            "Linear",
            ("relu",),
            spec=ParamSpec("dense", (4, 4)),
            meta={"shared_key": "m1"},
        ),
        Node("out", NodeKind.OUTPUT, "output", ("b",)),
    ]
    assert detect_shared(ModelGraph(nodes)) == {"m1": ("a", "b")}


def test_report_label_shows_the_merge_for_both_branches():
    """Check that the report shows the merge on both branches.

    Both normalizations in a transition block feed one post-addition activation, which is
    what distinguishes pairing by graph edges from pairing by layer order.
    """
    import anyinit

    torch = __import__("pytest").importorskip("torch")
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 8, 3, padding=1)
            self.bn = nn.BatchNorm2d(8)
            self.down = nn.Conv2d(4, 8, 1)
            self.down_bn = nn.BatchNorm2d(8)

        def forward(self, x):
            return torch.relu(self.bn(self.conv(x)) + self.down_bn(self.down(x)))

    report = anyinit.initialize(Block(), input_spec=(16, 4, 8, 8), seed=0)
    labels = {r.name: r.activation for r in report.layers if r.kind == "BatchNorm2d"}
    assert len(labels) == 2
    assert all(label == "add→relu" for label in labels.values()), labels


def test_report_label_shows_pooling_after_the_activation():
    """The target is enforced past the pooling, so the label says so."""
    import anyinit

    torch = __import__("pytest").importorskip("torch")
    nn = torch.nn

    class Stem(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.bn = nn.BatchNorm2d(8)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            return self.pool(torch.relu(self.bn(self.conv(x))))

    report = anyinit.initialize(Stem(), input_spec=(16, 3, 16, 16), seed=0)
    norm = next(r for r in report.layers if r.kind == "BatchNorm2d")
    assert norm.activation == "relu→MaxPool2d"
