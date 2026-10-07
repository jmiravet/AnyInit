"""Why a gain table is not enough: E[a^2] through a deep MLP, three ways.

A 20-layer, 256-wide, bias-free MLP per activation, initialized with the usual gain table
(He-style ``N(0, gain^2 / fan_in)``, reusing ReLU's gain for SiLU as is common), with
AnyInit's analytic mode and with its empirical mode.  E[a^2] is measured after every
activation on a fresh batch, independent of the one the empirical mode saw.  Prints the
mean and standard deviation over five seeds at layers 1, 10 and 20.

Needs PyTorch.  Quoted in README.md, "Why a gain table is not enough".
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

import anyinit

DEPTH, WIDTH, ROWS, SEEDS = 20, 256, 4096, range(5)
ACTIVATIONS: dict[str, tuple[type[nn.Module], float]] = {
    "ReLU": (nn.ReLU, math.sqrt(2.0)),
    "Tanh": (nn.Tanh, 5.0 / 3.0),
    "Sigmoid": (nn.Sigmoid, 1.0),
    "SiLU": (nn.SiLU, math.sqrt(2.0)),
}
LAYERS = (1, 10, 20)


def build(activation: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(DEPTH):
        layers += [nn.Linear(WIDTH, WIDTH, bias=False), activation()]
    return nn.Sequential(*layers)


def levels(model: nn.Sequential, batch: torch.Tensor) -> list[float]:
    """E[a^2] after every activation."""
    out = []
    with torch.no_grad():
        hidden = batch
        for index, layer in enumerate(model):
            hidden = layer(hidden)
            if index % 2 == 1:
                out.append(float(hidden.pow(2).mean()))
    return out


def run(name: str, method: str, seed: int) -> list[float]:
    activation, gain = ACTIVATIONS[name]
    torch.manual_seed(seed)
    model = build(activation)
    generator = torch.Generator().manual_seed(10_000 + seed)
    init_batch = torch.randn(ROWS, WIDTH, generator=generator)
    eval_batch = torch.randn(ROWS, WIDTH, generator=generator)
    if method == "gain table":
        with torch.no_grad():
            for layer in model:
                if isinstance(layer, nn.Linear):
                    layer.weight.normal_(0.0, gain / math.sqrt(WIDTH))
    elif method == "analytic":
        anyinit.initialize(model, "analytic", (ROWS, WIDTH), seed=seed)
    else:
        anyinit.initialize(model, "empirical", init_batch, seed=seed)
    return levels(model, eval_batch)


def main() -> None:
    print(f"E[a^2], {DEPTH}-layer {WIDTH}-wide MLP, mean +- std over {len(SEEDS)} seeds\n")
    print(f"{'activation':<10} {'method':<10}" + "".join(f"{f'layer {n}':>18}" for n in LAYERS))
    for name in ACTIVATIONS:
        for method in ("gain table", "analytic", "empirical"):
            runs = np.array([run(name, method, seed) for seed in SEEDS])
            cells = "".join(
                f"{runs[:, n - 1].mean():>10.3g} +- {runs[:, n - 1].std(ddof=1):<5.2g}"
                for n in LAYERS
            )
            print(f"{name:<10} {method:<10}{cells}")


if __name__ == "__main__":
    main()
