"""Gauss-Legendre against Gauss-Hermite and Monte Carlo, on a kinked function.

relu(x)**3 has closed-form moments, so the error of each rule is exact rather than
estimated: E[f] = sqrt(2/pi) and E[f^2] = 15/2.
"""

from __future__ import annotations

import math

import numpy as np

from anyinit.core.quadrature import detect_kinks, gauss_legendre_nodes, integrate

RELU3 = lambda x: np.maximum(x, 0.0) ** 3  # noqa: E731
EXACT_M1 = math.sqrt(2.0 / math.pi)
EXACT_M2 = 7.5


def gauss_legendre(panels: int, nodes: int) -> tuple[int, float, float]:
    kinks = detect_kinks(RELU3, -10.0, 10.0)
    z, w = gauss_legendre_nodes(0.0, 1.0, panels=panels, nodes=nodes, kinks=kinks)
    m1, m2 = integrate(RELU3(z), w, (1, 2))
    return z.size, abs(m1 - EXACT_M1), abs(m2 - EXACT_M2)


def gauss_hermite(count: int) -> tuple[int, float, float]:
    t, w = np.polynomial.hermite.hermgauss(count)
    z = t * math.sqrt(2.0)
    weights = w / w.sum()
    values = RELU3(z)
    m1 = float((weights * values).sum())
    m2 = float((weights * values**2).sum())
    return count, abs(m1 - EXACT_M1), abs(m2 - EXACT_M2)


def monte_carlo(count: int, seed: int = 0) -> tuple[int, float, float]:
    sample = RELU3(np.random.default_rng(seed).standard_normal(count))
    return count, abs(sample.mean() - EXACT_M1), abs((sample**2).mean() - EXACT_M2)


def main() -> None:
    rows = [
        ("Gauss-Legendre 8x16", *gauss_legendre(8, 16)),
        ("Gauss-Legendre 32x16", *gauss_legendre(32, 16)),
        ("Gauss-Hermite 64", *gauss_hermite(64)),
        ("Gauss-Hermite 256", *gauss_hermite(256)),
        ("Monte Carlo 1e6", *monte_carlo(10**6)),
        ("Monte Carlo 1e7", *monte_carlo(10**7)),
    ]
    print(f"{'rule':24s} {'evaluations':>12s} {'err E[f]':>11s} {'err E[f^2]':>11s}")
    for name, count, e1, e2 in rows:
        print(f"{name:24s} {count:12d} {e1:11.2e} {e2:11.2e}")

    print(
        "\nGauss-Hermite reaches E[f^2] exactly by symmetry -- relu(x)^6 + relu(-x)^6 is a\n"
        "degree-six polynomial -- but stalls on E[f] because of the kink at zero.  Composite\n"
        "Gauss-Legendre puts a panel boundary there and is exact in both."
    )


if __name__ == "__main__":
    main()
