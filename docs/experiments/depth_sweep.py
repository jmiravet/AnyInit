"""Which activations survive depth, measured rather than predicted.

Rescales each layer from measurements so that E[a^2] is one after every activation -- the
best any per-layer scalar can do -- and reports what actually arrives at the end.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

DEPTHS_AND_WIDTHS = ((3, 256), (5, 256), (10, 256), (20, 256), (20, 1024), (20, 4096))


def network(power: int, depth: int, width: int) -> tuple[nn.Module, type[nn.Module]]:
    class Power(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x) ** power

    layers: list[nn.Module] = []
    for _ in range(depth):
        layers += [nn.Linear(width, width, bias=False), Power()]
    return nn.Sequential(*layers), Power


def rescale(model: nn.Module, activation: type[nn.Module], power: int, rows: int) -> None:
    """Drive E[a^2] to one after every activation, measuring as we go."""
    first = next(m for m in model if isinstance(m, nn.Linear))
    hidden = torch.randn(rows, first.in_features)
    with torch.no_grad():
        for index, layer in enumerate(model):
            if isinstance(layer, nn.Linear):
                follower = model[index + 1]
                for _ in range(40):
                    value = float(follower(layer(hidden)).pow(2).mean())
                    if not math.isfinite(value) or value <= 0.0:
                        layer.weight.mul_(0.5)
                        continue
                    layer.weight.mul_((1.0 / value) ** (1.0 / (2 * power)))
                    if abs(value - 1.0) < 1e-5:
                        break
            hidden = layer(hidden)


def main(rows: int = 2048) -> None:
    print(f"{'p':>3s} {'depth':>6s} {'width':>6s} {'final E[a^2]':>13s} {'top-8 energy':>13s}")
    for power in (1, 2, 3):
        for depth, width in DEPTHS_AND_WIDTHS:
            torch.manual_seed(0)
            model, activation = network(power, depth, width)
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0.0, 0.05)
            rescale(model, activation, power, rows)

            with torch.no_grad():
                output = model(torch.randn(rows, width))
            energy = output.pow(2)
            total = float(energy.sum())
            share = (
                float(energy.sum(0).sort(descending=True).values[:8].sum()) / total
                if total > 0.0
                else float("nan")
            )
            print(
                f"{power:3d} {depth:6d} {width:6d} {float(energy.mean()):13.4g} "
                f"{share * 100:12.1f}%"
            )

    print(
        "\nDegree one is stable at any depth and widening only spreads the energy further.\n"
        "Above it no amount of per-layer rescaling holds: the signal concentrates into a\n"
        "handful of units and then dies, and extra width does not help."
    )


if __name__ == "__main__":
    main()
