"""How far one draw lands from the analytic prediction, and what removes the gap.

A 20-layer ReLU MLP, initialized over 16 seeds three ways: analytic, analytic with
``center=True``, and empirical.  Prints the final layer's E[a^2] (target 1) across seeds.
The analytic mode predicts the ensemble; any one draw of a deep, narrow network spreads
around it, and the spread shrinks with width.

Needs PyTorch.  Quoted in README.md, "Validity".
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import anyinit

DEPTH = 20
SEEDS = 16


def final_level(width: int, mode: str, seed: int, **options: object) -> float:
    layers: list[nn.Module] = []
    fan_in = 128
    for _ in range(DEPTH):
        layers += [nn.Linear(fan_in, width), nn.ReLU()]
        fan_in = width
    model = nn.Sequential(*layers)
    batch = torch.randn(4096, 128, generator=torch.Generator().manual_seed(0))
    anyinit.initialize(model, mode, batch, seed=seed, **options)
    with torch.no_grad():
        return float(model(batch).pow(2).mean())


def main() -> None:
    print(f"final E[a^2] of a {DEPTH}-layer ReLU MLP over {SEEDS} seeds (target 1)\n")
    print(f"{'width':>5}  {'mode':<18} {'median':>6}  {'range':<14}")
    for width in (128, 512):
        for label, mode, options in (
            ("analytic", "analytic", {}),
            ("analytic, center", "analytic", {"center": True}),
            ("empirical", "empirical", {}),
        ):
            levels = np.array([final_level(width, mode, s, **options) for s in range(SEEDS)])
            spread = f"[{levels.min():.2f}, {levels.max():.2f}]"
            print(f"{width:>5}  {label:<18} {np.median(levels):6.2f}  {spread:<14}")


if __name__ == "__main__":
    main()
