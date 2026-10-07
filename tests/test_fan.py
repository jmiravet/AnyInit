"""Fan arithmetic, including the cases frameworks get wrong."""

from __future__ import annotations

import pytest

from anyinit.core.fan import ParamSpec, fan_in


def test_dense():
    spec = ParamSpec("dense", (128, 256))
    assert fan_in(spec) == 256


def test_conv():
    spec = ParamSpec("conv", (32, 16, 3, 3))
    assert fan_in(spec) == 16 * 9


def test_depthwise_fan_in_is_the_kernel():
    """Each output channel of a depthwise convolution reads one input channel."""
    spec = ParamSpec("conv", (16, 1, 3, 3), groups=16)
    assert fan_in(spec) == 9


def test_grouped_conv_fan_in_is_per_group():
    spec = ParamSpec("conv", (32, 8, 3, 3), groups=4)
    assert fan_in(spec) == 8 * 9


def test_transposed_conv_canonical_shape_gives_the_operations_fan():
    """Check that the canonical shape gives the operation's own fan.

    A torch ConvTranspose2d(16, 32, 3) stores (16, 32, 3, 3), whose naive reading is 288
    where the operation's fan_in is 144.
    """
    spec = ParamSpec("conv_transpose", (32, 16, 3, 3))
    assert fan_in(spec) == 144


def test_packed_attention_fan_in_is_the_embedding():
    """A packed (3E, E) projection reads E inputs per output, like each of its blocks."""
    assert fan_in(ParamSpec("attention", (192, 64))) == 64


def test_embedding_is_not_fan_scaled():
    assert not ParamSpec("embedding", (30000, 512)).is_fan_scaled


def test_norm_is_not_fan_scaled():
    assert not ParamSpec("norm", (64,)).is_fan_scaled


def test_rejects_bad_groups():
    with pytest.raises(ValueError, match="groups"):
        ParamSpec("conv", (4, 4, 1, 1), groups=0)


def test_rejects_underspecified_shape():
    with pytest.raises(ValueError, match="2-D canonical shape"):
        ParamSpec("dense", (4,))
