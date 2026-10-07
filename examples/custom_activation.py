"""Teaching AnyInit an activation it has never seen.

Run: python examples/custom_activation.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import anyinit


@anyinit.register_activation
class ReLUCubed(nn.Module):
    """x -> relu(x)**3.

    Registered as a module rather than a function because torch.fx then keeps it as a
    single graph node instead of tracing into it as a relu and a pow.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) ** 3


def show_profile() -> None:
    profile = anyinit.activation_profile("relucubed")
    state = profile.moments(0.0, 1.0)

    print("Profile of relu(x)**3, computed by quadrature:")
    print(f"  E[f]   = {state.mean:.10f}   (exact: sqrt(2/pi) = {np.sqrt(2 / np.pi):.10f})")
    print(f"  E[f^2] = {state.m2:.10f}   (exact: 15/2 = 7.5)")
    print(f"  degree = {profile.homogeneous_degree}")
    print(f"  chi    = {profile.chi:.6f}")
    print(f"  sigma* = {profile.diagnostics.sigma_star:.6f}")


def initialize_a_network(depth: int) -> None:
    torch.manual_seed(0)
    layers: list[nn.Module] = []
    for _ in range(depth):
        layers += [nn.Linear(128, 128, bias=False), ReLUCubed()]
    model = nn.Sequential(*layers)

    report = anyinit.initialize(model, input_spec=(1024, 128), seed=0)

    with torch.no_grad():
        hidden = torch.randn(4096, 128)
        levels = []
        for layer in model:
            hidden = layer(hidden)
            if isinstance(layer, ReLUCubed):
                levels.append(float(hidden.pow(2).mean()))

    print(f"\nDepth {depth}: E[a^2] after each activation")
    print("  " + "  ".join(f"{value:.3g}" for value in levels[:6]) + (" ..." if depth > 6 else ""))
    for record in report.stability:
        advice = record.advice()
        if advice:
            print(f"  ! {advice}")


if __name__ == "__main__":
    show_profile()
    # Shallow is fine; depth is not, and the report says why rather than leaving you to
    # discover it from a loss that never moves.
    initialize_a_network(depth=2)
    initialize_a_network(depth=20)
