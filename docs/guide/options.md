# Options

The defaults work for most models. Every argument:

```python
anyinit.initialize(
    model,
    mode="analytic",  # analytic | empirical
    input_spec=(32, 64),  # a shape, a batch, or a callable returning one
    distribution="normal",  # normal | uniform | sinusoidal
    center=False,
    gains={"relu": 2**0.5},  # fix the gain for an activation instead of solving it
    seed=0,
)
```

## `mode`

Where the moments come from: theory (`analytic`) or real batches (`empirical`). See
[Analytic or empirical](modes.md).

## `input_spec`

A shape, a batch, or a callable returning one. `analytic` uses only its shape, and runs a
validation pass when it is given; `empirical` needs real data.

## `center`

`center=True` subtracts each output unit's mean weight, so a layer discards its input's
mean instead of passing it on. This makes a single draw land much closer to the
prediction in deep networks, at the cost of one input direction per layer. It applies
only to layers wide enough to spare it.

## `gains`

Maps an activation name to a fixed gain. Layers feeding that activation get
`gain / √fan_in` (a normalization gets `gain` itself) and are left out of the solve; the
rest are still solved around them. To see the gain AnyInit would choose:

```python
anyinit.gain("relu")  # 1.4142…, √2
anyinit.gain("silu")  # 1.5588…
anyinit.gain("leaky_relu", negative_slope=0.2)
```

## `distribution`

The weight distribution: `normal`, `uniform`, or `sinusoidal`. With `sinusoidal`, prefer
the `empirical` mode.

## `seed`

Weights are drawn from it in NumPy, so the same seed gives the same weights in every
framework. Biases start at zero.
