"""The chi tables: which activations can hold a signal across depth.

Prints, as Markdown, the stability table in README.md and the moment table in
docs/reference.md.  tests/test_docs_tables.py checks that both still match this output.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from anyinit.core.activations import BUILTIN
from anyinit.core.profile import ActivationProfile
from anyinit.core.stability import DRIFT_LIMIT, stable_depth

Fn = Callable[[np.ndarray], np.ndarray]


def _power(p: int) -> Fn:
    return lambda x: np.maximum(x, 0.0) ** p


#: Display name, function.  The order is the order of both tables.
ROWS: list[tuple[str, Fn]] = [
    ("relu", BUILTIN["relu"]),
    ("relu²", _power(2)),
    ("relu³", _power(3)),
    *((name, BUILTIN[name]) for name in ("gelu", "silu", "elu", "selu", "mish", "softplus")),
    ("tanh", BUILTIN["tanh"]),
    ("sigmoid", BUILTIN["sigmoid"]),
]


def profile(name: str, fn: Fn) -> ActivationProfile:
    return ActivationProfile(name, fn)


def depth_column(chi: float, verdict: str) -> str:
    """Layers before a relative error grows ``DRIFT_LIMIT``-fold."""
    if verdict == "infeasible":
        return "—"
    depth = stable_depth(chi)
    return "any" if math.isinf(depth) else f"{math.floor(depth)}"


def stability_table() -> list[str]:
    lines = [
        f"| activation | χ | verdict | depth before an error grows {DRIFT_LIMIT:g}× |",
        "|---|---|---|---|",
    ]
    for name, fn in ROWS:
        diag = profile(name, fn).diagnostics
        depth = depth_column(diag.chi, diag.verdict)
        lines.append(f"| `{name}` | {diag.chi:.3f} | {diag.verdict} | {depth} |")
    return lines


def moment_table() -> list[str]:
    lines = [
        "| `f` | `E[f]` | `E[f²]` | gain | `χ` | `σ*` | degree |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, fn in ROWS:
        prof = profile(name, fn)
        state = prof.moments(0.0, 1.0)
        diag = prof.diagnostics
        star = "unreachable" if diag.sigma_star is None else f"{diag.sigma_star:.4f}"
        degree = "—" if diag.homogeneous_degree is None else f"{diag.homogeneous_degree:g}"
        lines.append(
            f"| `{name}` | {state.mean:.7f} | {state.m2:.7f} | {1.0 / math.sqrt(state.m2):.7f} "
            f"| {diag.chi:.3f} | {star} | {degree} |"
        )
    return lines


def main() -> None:
    print("README.md, stability:\n")
    print("\n".join(stability_table()))
    print("\ndocs/reference.md, moments:\n")
    print("\n".join(moment_table()))


if __name__ == "__main__":
    main()
