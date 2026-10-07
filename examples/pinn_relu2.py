"""A physics-informed network with relu(x)**2, and an honest result.

Solves u'' = f on [0, 1] with u(0) = u(1) = 0, imposing the boundary condition by
construction.  The activation is homogeneous of degree two, which AnyInit reports as
chi = 2: not depth-stable, usable at the three layers this network has.

What AnyInit contributes here is the diagnosis, not a training win.  It tells you up
front that relu(x)**2 cannot survive depth, which is worth knowing before you make the
network deeper and wonder why it stops training.

The training numbers are printed but should not be read as evidence either way.  A PINN
whose loss is on a second derivative is badly conditioned, and sweeping the learning rate
gives neither initialization a monotone curve -- PyTorch's best is 0.295 at 2e-3 while
AnyInit's is 0.898 at 5e-5, with both far worse in between.  A single-seed comparison on
this problem measures the learning rate, not the initialization.  See
examples/depth_matters.py for a setting where the initialization is what decides the
outcome.

Run: python examples/pinn_relu2.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import anyinit


@anyinit.register_activation
class ReLUSquared(nn.Module):
    """x -> relu(x)**2, homogeneous of degree two."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) ** 2


class PINN(nn.Module):
    """Three hidden layers, each feeding the output through its own skip."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.layer1 = nn.Linear(1, width)
        self.layer2 = nn.Linear(width, width)
        self.layer3 = nn.Linear(width, width)
        self.act = ReLUSquared()
        self.skip1 = nn.Linear(width, 1, bias=False)
        self.skip2 = nn.Linear(width, 1, bias=False)
        self.skip3 = nn.Linear(width, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act(self.layer1(x))
        h2 = self.act(self.layer2(h1))
        h3 = self.act(self.layer3(h2))
        raw = self.skip1(h1) + self.skip2(h2) + self.skip3(h3)
        return x * (1.0 - x) * raw  # u(0) = u(1) = 0 by construction


def forcing(x: torch.Tensor, m: int, n: int) -> torch.Tensor:
    return -((2 * math.pi * m) ** 2) * torch.sin(2 * math.pi * m * x) - (
        (2 * math.pi * n) ** 2
    ) * torch.sin(2 * math.pi * n * x)


def true_solution(x: torch.Tensor, m: int, n: int) -> torch.Tensor:
    return torch.sin(2 * math.pi * m * x) + torch.sin(2 * math.pi * n * x)


def residual_loss(model: nn.Module, x: torch.Tensor, m: int, n: int) -> torch.Tensor:
    x = x.requires_grad_(True)
    u = model(x)
    (du,) = torch.autograd.grad(u.sum(), x, create_graph=True)
    (d2u,) = torch.autograd.grad(du.sum(), x, create_graph=True)
    return (d2u - forcing(x, m, n)).pow(2).mean()


def train(initialize: bool, steps: int = 400, m: int = 1, n: int = 4) -> float:
    torch.manual_seed(0)
    model = PINN()
    if initialize:
        report = anyinit.initialize(model, input_spec=(512, 1), seed=0)
        if initialize == "verbose":
            print(report)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    grid = torch.linspace(0.0, 1.0, 512).unsqueeze(1)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = residual_loss(model, grid.clone(), m, n)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test = torch.linspace(0.0, 1.0, 1000).unsqueeze(1)
        error = (model(test) - true_solution(test, m, n)).pow(2).mean().sqrt()
    return float(error)


if __name__ == "__main__":
    profile = anyinit.activation_profile("relusquared")
    print("What AnyInit can tell you about relu(x)**2 before you train anything:")
    print(f"  E[f^2]  = {profile.moments(0.0, 1.0).m2:.6f}   (exact: 3/2)")
    print(f"  degree  = {profile.homogeneous_degree:g}")
    print(f"  chi     = {profile.chi:.4f}")
    print(f"  sigma*  = {profile.diagnostics.sigma_star:.4f}")
    print(
        "  chi = 2 means a relative error in the variance doubles every layer, so no scalar\n"
        "  initialization is depth-stable.  At three layers that does not bite.\n"
    )

    baseline = train(initialize=False)
    tuned = train(initialize=True)
    print(f"  PyTorch default init:  final RMSE {baseline:.4f}")
    print(f"  AnyInit:              final RMSE {tuned:.4f}")
    print(
        "\n  Read those two numbers with care: a PINN loss on u'' is badly conditioned,\n"
        "  and neither initialization has a monotone learning-rate curve on this problem,\n"
        "  so a single-seed comparison measures the learning rate rather than the\n"
        "  initialization.  examples/depth_matters.py is the setting where the\n"
        "  initialization is what decides the outcome."
    )
