"""Canonical parameter shapes and the fan arithmetic built on them.

Each framework stores convolution kernels in its own axis order, and a transposed
convolution swaps the input and output axes relative to an ordinary one.  AnyInit defines
a single canonical layout, ``(fan_out_axis, fan_in_axis, *receptive_field)``, and leaves
the translation to each backend, so the arithmetic lives here once and accounts for
``groups``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Layer families AnyInit knows how to scale.
DENSE = "dense"
CONV = "conv"
CONV_TRANSPOSE = "conv_transpose"
EMBEDDING = "embedding"
ATTENTION = "attention"
NORM = "norm"

#: Families whose scale follows from fan arithmetic.  ``embedding`` is excluded: a lookup
#: is a product with a one-hot vector, so there is no sum over fan_in to compensate for.
FAN_SCALED = frozenset({DENSE, CONV, CONV_TRANSPOSE, ATTENTION})


@dataclass(frozen=True)
class ParamSpec:
    """Shape and wiring of one scalable parameter block, in canonical layout."""

    kind: str
    canonical_shape: tuple[int, ...]
    groups: int = 1
    has_bias: bool = False

    def __post_init__(self) -> None:
        if self.groups < 1:
            raise ValueError(f"groups must be >= 1, got {self.groups}")
        if self.kind in FAN_SCALED and len(self.canonical_shape) < 2:
            raise ValueError(
                f"{self.kind} expects at least a 2-D canonical shape, got {self.canonical_shape}"
            )

    @property
    def receptive(self) -> int:
        """Number of spatial positions each output unit reads per input channel."""
        return int(math.prod(self.canonical_shape[2:])) if len(self.canonical_shape) > 2 else 1

    @property
    def out_units(self) -> int:
        """Size of the output axis."""
        return int(self.canonical_shape[0])

    @property
    def in_units(self) -> int:
        """Size of the input axis, already divided by ``groups``."""
        return int(self.canonical_shape[1]) if len(self.canonical_shape) > 1 else 1

    @property
    def size(self) -> int:
        """Total number of elements."""
        return int(math.prod(self.canonical_shape))

    @property
    def is_fan_scaled(self) -> bool:
        """Whether this family's scale follows from fan arithmetic."""
        return self.kind in FAN_SCALED


def fan_in(spec: ParamSpec) -> float:
    """Return the number of inputs summed into one output unit.

    The canonical in-axis already holds ``in_channels / groups``.
    """
    return float(spec.in_units * spec.receptive)
