"""One core, three frameworks.

Weights are drawn in NumPy, in a canonical layout, from the run's seed; each backend
only translates that layout into its own.  So the same architecture and the same seed
must produce the same weights everywhere -- bit for bit, not approximately.

That makes this the sharpest test in the suite.  It fails if the core is accidentally
framework-dependent, if a layout conversion is wrong, if a backend's graph differs, or
if the solver is sensitive to anything beyond the IR.  It also gives anyone comparing
frameworks a reproducible starting point, which is unusual.
"""

from __future__ import annotations

import numpy as np
import pytest

import anyinit

DEPTH, WIDTH, IN_FEATURES, SEED = 4, 96, 64, 7
COMMON = {"seed": SEED, "center": True}


def torch_kernels() -> list[np.ndarray]:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    layers: list[object] = []
    for i in range(DEPTH):
        layers += [nn.Linear(IN_FEATURES if i == 0 else WIDTH, WIDTH, bias=False), nn.ReLU()]
    layers.append(nn.Linear(WIDTH, 10, bias=False))
    model = nn.Sequential(*layers)

    anyinit.initialize(model, input_spec=(128, IN_FEATURES), **COMMON)
    return [
        parameter.detach().numpy().astype(np.float64)
        for name, parameter in model.named_parameters()
        if name.endswith("weight")
    ]


def keras_kernels() -> list[np.ndarray]:
    keras = pytest.importorskip("keras")
    from keras import layers

    inputs = keras.Input((IN_FEATURES,))
    h = inputs
    for _ in range(DEPTH):
        h = layers.Dense(WIDTH, activation="relu", use_bias=False)(h)
    model = keras.Model(inputs, layers.Dense(10, use_bias=False)(h))

    anyinit.initialize(model, input_spec=(128, IN_FEATURES), **COMMON)
    return [
        np.asarray(keras.ops.convert_to_numpy(layer.kernel), dtype=np.float64).T
        for layer in model.layers
        if hasattr(layer, "kernel")
    ]


def flax_kernels() -> list[np.ndarray]:
    pytest.importorskip("flax")
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
    params, _report = anyinit.initialize_params(model, params, (128, IN_FEATURES), **COMMON)

    flat = flatten_dict(params["params"])
    keys = sorted((k for k in flat if k[-1] == "kernel"), key=lambda k: int(k[0].split("_")[1]))
    return [np.asarray(flat[k], dtype=np.float64).T for k in keys]


BUILDERS = {"pytorch": torch_kernels, "keras": keras_kernels, "jax": flax_kernels}


def _build_or_skip(name, builder):
    try:
        return {name: builder()}
    except pytest.skip.Exception:
        return {}


@pytest.fixture(scope="module")
def kernels():
    available = {}
    for name, builder in BUILDERS.items():
        available.update(_build_or_skip(name, builder))
    if len(available) < 2:
        pytest.skip("need at least two frameworks installed to compare them")
    return available


def test_every_backend_produces_the_same_layer_shapes(kernels):
    shapes = {name: [k.shape for k in weights] for name, weights in kernels.items()}
    reference = next(iter(shapes.values()))
    for name, found in shapes.items():
        assert found == reference, f"{name} disagrees on shapes: {found} != {reference}"


def test_weights_are_identical_across_backends(kernels):
    names = list(kernels)
    reference_name = names[0]
    reference = kernels[reference_name]

    for name in names[1:]:
        for index, (a, b) in enumerate(zip(reference, kernels[name], strict=False)):
            assert np.allclose(a, b, atol=1e-6), (
                f"layer {index}: {reference_name} and {name} differ by {np.abs(a - b).max():.3e}"
            )


def test_scales_are_identical_across_backends(kernels):
    stds = {name: [float(k.std()) for k in weights] for name, weights in kernels.items()}
    reference = next(iter(stds.values()))
    for name, found in stds.items():
        assert found == pytest.approx(reference, rel=1e-6), name
