"""The chi tables: which activations can hold a signal across depth.

Prints, as Markdown, the stability table in docs/guide/stability.md and the moment table in
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


def fixed(value: float, digits: int) -> str:
    """``value`` to ``digits`` decimals, with no sign on a zero.

    A mean that is zero in exact arithmetic comes out of the quadrature as ±1e-17, its sign
    set by the platform's libm, and would otherwise print as ``-0.0000000`` on some of them.
    """
    return f"{round(value, digits) + 0.0:.{digits}f}"


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
        lines.append(f"| `{name}` | {fixed(diag.chi, 3)} | {diag.verdict} | {depth} |")
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
        star = "unreachable" if diag.sigma_star is None else fixed(diag.sigma_star, 4)
        degree = "—" if diag.homogeneous_degree is None else f"{diag.homogeneous_degree:g}"
        lines.append(
            f"| `{name}` | {fixed(state.mean, 7)} | {fixed(state.m2, 7)} "
            f"| {fixed(1.0 / math.sqrt(state.m2), 7)} | {fixed(diag.chi, 3)} | {star} | {degree} |"
        )
    return lines


def main() -> None:
    print("docs/guide/stability.md, stability:\n")
    print("\n".join(stability_table()))
    print("\ndocs/reference.md, moments:\n")
    print("\n".join(moment_table()))


if __name__ == "__main__":
    main()
