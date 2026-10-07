"""Model builders shared by the behavioral tests, one per framework."""

from __future__ import annotations

from typing import Any

import pytest

ACTIVATIONS = ["relu", "gelu", "silu", "elu", "tanh", "sigmoid", "mish", "softplus"]
#: Activations whose forward second moment is reachable, so a target of 1 applies.
UNBOUNDED = ["relu", "gelu", "silu", "elu", "mish", "softplus"]


def torch_mlp(depth: int = 20, width: int = 256, activation: str = "relu", in_features: int = 256):
    torch = pytest.importorskip("torch")
    nn = torch.nn
    table = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "mish": nn.Mish,
        "softplus": nn.Softplus,
    }
    layers: list[Any] = []
    for i in range(depth):
        layers.append(nn.Linear(in_features if i == 0 else width, width, bias=False))
        layers.append(table[activation]())
    return nn.Sequential(*layers), table[activation]


def torch_activation_moments(model: Any, activation_type: Any, rows: int = 4096):
    """E[a^2] after every activation, measured on a fresh batch."""
    torch = pytest.importorskip("torch")
    first = next(m for m in model if hasattr(m, "in_features"))
    h = torch.randn(rows, first.in_features)
    out = []
    with torch.no_grad():
        for layer in model:
            h = layer(h)
            if isinstance(layer, activation_type):
                out.append(float(h.pow(2).mean()))
    return out


def keras_mlp(depth: int = 4, width: int = 96, activation: str = "relu", in_features: int = 64):
    keras = pytest.importorskip("keras")
    from keras import layers

    inputs = keras.Input((in_features,))
    h = inputs
    for _ in range(depth):
        h = layers.Dense(width, activation=activation, use_bias=False)(h)
    return keras.Model(inputs, layers.Dense(10, use_bias=False)(h))


def flax_mlp(depth: int = 4, width: int = 96, activation: str = "relu", in_features: int = 64):
    pytest.importorskip("flax")
    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    act = getattr(jax.nn, activation)

    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            for _ in range(depth):
                x = act(nn.Dense(width, use_bias=False)(x))
            return nn.Dense(10, use_bias=False)(x)

    model = MLP()
    params = model.init(jax.random.key(0), jnp.ones((1, in_features)))
    return model, params
