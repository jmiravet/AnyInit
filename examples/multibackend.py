"""The same network, the same seed, three frameworks.

Weights are drawn in NumPy from the run's seed in a canonical layout, and each backend
only translates that layout into its own.  So the three frameworks end up with the same
numbers -- which makes a cross-framework comparison reproducible, and is a strong check
that nothing framework-specific leaked into the solver.

Run: python examples/multibackend.py
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import anyinit

DEPTH, WIDTH, IN_FEATURES, SEED = 4, 96, 64, 7
SETTINGS = {"seed": SEED, "center": True}


def with_pytorch() -> list[np.ndarray]:
    import torch.nn as nn

    layers: list[nn.Module] = []
    for i in range(DEPTH):
        layers += [nn.Linear(IN_FEATURES if i == 0 else WIDTH, WIDTH, bias=False), nn.ReLU()]
    layers.append(nn.Linear(WIDTH, 10, bias=False))
    model = nn.Sequential(*layers)

    anyinit.initialize(model, input_spec=(128, IN_FEATURES), **SETTINGS)
    return [
        p.detach().numpy().astype(np.float64)
        for name, p in model.named_parameters()
        if name.endswith("weight")
    ]


def with_keras() -> list[np.ndarray]:
    import keras
    from keras import layers

    inputs = keras.Input((IN_FEATURES,))
    hidden = inputs
    for _ in range(DEPTH):
        hidden = layers.Dense(WIDTH, activation="relu", use_bias=False)(hidden)
    model = keras.Model(inputs, layers.Dense(10, use_bias=False)(hidden))

    anyinit.initialize(model, input_spec=(128, IN_FEATURES), **SETTINGS)
    return [
        np.asarray(keras.ops.convert_to_numpy(layer.kernel), dtype=np.float64).T
        for layer in model.layers
        if hasattr(layer, "kernel")
    ]


def with_flax() -> list[np.ndarray]:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict

    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            for _ in range(DEPTH):
                x = jax.nn.relu(nn.Dense(WIDTH, use_bias=False)(x))
            return nn.Dense(10, use_bias=False)(x)

    model = MLP()
    params = model.init(jax.random.key(0), jnp.ones((1, IN_FEATURES)))
    params, _report = anyinit.initialize_params(model, params, (128, IN_FEATURES), **SETTINGS)

    flat = flatten_dict(params["params"])
    keys = sorted((k for k in flat if k[-1] == "kernel"), key=lambda k: int(k[0].split("_")[1]))
    return [np.asarray(flat[k], dtype=np.float64).T for k in keys]


def _try(name: str, builder: Callable[[], list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
    """Run one builder, reporting rather than failing when its framework is absent."""
    try:
        return {name: builder()}
    except ImportError:
        print(f"{name:10s} not installed, skipping")
        return {}


def main() -> None:
    builders = {"pytorch": with_pytorch, "keras": with_keras, "flax": with_flax}
    results: dict[str, list[np.ndarray]] = {}

    for name, builder in builders.items():
        results.update(_try(name, builder))

    if len(results) < 2:
        print("\nInstall at least two frameworks to see the comparison.")
        return

    reference_name, reference = next(iter(results.items()))
    print(f"\nComparing against {reference_name}, seed={SEED}\n")
    header = f"{'layer':>6s} {'shape':>12s} " + " ".join(
        f"{n:>14s}" for n in results if n != reference_name
    )
    print(header)
    print("-" * len(header))

    for index, weight in enumerate(reference):
        gaps = [
            f"{np.abs(weight - results[name][index]).max():14.2e}"
            for name in results
            if name != reference_name
        ]
        print(f"{index:6d} {weight.shape!s:>12s} " + " ".join(gaps))

    print("\nColumns are the largest absolute difference per layer.")


if __name__ == "__main__":
    main()
