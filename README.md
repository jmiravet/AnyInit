# AnyInit

[![CI](https://github.com/jmiravet/AnyInit/actions/workflows/ci.yml/badge.svg)](https://github.com/jmiravet/AnyInit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jmiravet/AnyInit/graph/badge.svg)](https://codecov.io/gh/jmiravet/AnyInit)
[![PyPI](https://img.shields.io/pypi/v/anyinit)](https://pypi.org/project/anyinit/)
[![Python](https://img.shields.io/pypi/pyversions/anyinit)](https://pypi.org/project/anyinit/)
[![Docs](https://img.shields.io/badge/docs-jmiravet.github.io%2FAnyInit-blue)](https://jmiravet.github.io/AnyInit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/jmiravet/AnyInit/blob/main/LICENSE)

Initialize any model, in any framework, correctly — with one call.

```python
import anyinit

report = anyinit.initialize(model)
```

`model` can be a `torch.nn.Module`, a Keras model, or a Flax module. AnyInit works out
which framework it is, traces the architecture, figures out which activation follows each
layer, and scales every weight so the signal neither dies nor explodes on its way through.
Nothing to configure, nothing to look up.

## What "any" means

**Any framework.** PyTorch, Keras 3 (on TensorFlow, JAX or PyTorch) and JAX (via Flax),
detected from the model itself.

```bash
pip install anyinit
```

That is the only install. AnyInit depends on NumPy alone and picks up whichever framework
your model comes from, including one installed afterward; the others are never imported.
To see what is usable here, without importing anything:

```python
anyinit.available_backends()   # ('pytorch',)
anyinit.known_backends()       # ('keras', 'jax', 'pytorch')
```

**Any activation.** Not a table of known names. AnyInit measures whatever function you
hand it, so an activation it has never seen is a first-class input:

```python
import torch, anyinit

@anyinit.register_activation
class ReLUCubed(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) ** 3

anyinit.initialize(model)      # relu³ is now scaled as precisely as relu
```

Its moment map is computed by Gauss–Legendre quadrature over the Gaussian: exact to
machine precision, even across the kink, in ~128 function evaluations. For `relu³` that
recovers `E[f] = 0.7978846` and `E[f²] = 7.5` to twelve digits, where 10⁷ Monte Carlo
samples still carry ~1e-3 error.

Write it against NumPy and it works with every backend without importing any framework:

```python
anyinit.register_activation(lambda x: np.maximum(x, 0) ** 3, name="relu3")
```

**Any architecture.** Branches, residual additions, concatenations, normalization layers,
pooling, dropout, attention, embeddings. Layers are paired with the activations they feed
by following graph edges, so both normalizations in a ResNet transition block are
recognized as feeding the same post-addition ReLU, and scaled together.

**Either theory or measurement.** Two modes, one argument apart:

```python
anyinit.initialize(model, "analytic", (32, 3, 224, 224))   # no data, no forward pass
anyinit.initialize(model, "empirical", batch)              # no assumptions
```

`analytic` propagates moments through the graph and never runs the model — `input_spec`
contributes shapes and nothing else. Deterministic, data-free, milliseconds.

`empirical` pushes real batches through, measures what arrives, and corrects layer by
layer in topological order, which makes each one exact given everything upstream. Reach
for it when the analytic mode warns about its own assumptions.

Both share the graph, the layer adapters and the target policy, so switching between them
changes the method and not the goal.

## Why a gain table is not enough

The usual recipe picks a per-activation constant — `√2` for ReLU, `5/3` for tanh — and
scales each layer by `gain / √fan_in`. That is correct only when the activation is
positively homogeneous of degree one. Everything else has a gain that depends on the
variance actually arriving, and the error compounds with depth.

Measured `E[a²]` after each activation of a 20-layer, 256-wide MLP, mean ± standard
deviation over five seeds (target 1):

| activation | gain table, layer 1 | layer 10 | layer 20 | AnyInit analytic, layer 20 | AnyInit empirical, layer 20 |
|---|---|---|---|---|---|
| ReLU | 1.00 ± 0.01 | 1.21 ± 0.58 | 1.20 ± 0.33 | 0.81 ± 0.35 | 1.00 ± 0.00 |
| Tanh | 0.58 ± 0.00 | 0.42 ± 0.00 | 0.42 ± 0.00 | 0.50 ± 0.00 \* | 0.50 ± 0.00 \* |
| Sigmoid | 0.29 ± 0.00 | 0.27 ± 0.01 | 0.27 ± 0.01 | 0.38 ± 0.01 \* | 0.38 ± 0.01 \* |
| SiLU | 0.80 ± 0.01 | 0.019 ± 0.009 | **2.0e-05** | 3.2 ± 1.9 † | 1.02 ± 0.03 |

ReLU is the case the table was derived for, and it holds. SiLU borrows ReLU's gain, as is
common practice, and the signal vanishes.

\* tanh and sigmoid are bounded, so `E[a²] = 1` is unreachable; AnyInit targets the middle
of the range each can reach and says so in the report.

† SiLU's `χ = 1.147` makes it unstable at this depth: an error grows `1.147²⁰ ≈ 15`-fold
over 20 layers, so the analytic mode's ensemble prediction is not held by any one draw. The
report says so; the empirical mode corrects the draw.

Reproduce with `python docs/experiments/gain_table.py`.

A gain is a constant. The correct object is a **map**: for any `f`, the moments of `f(z)`
when `z ~ N(μ, σ²)`. AnyInit computes that map, propagates it through the model's graph,
and solves for one scale per layer.

## It tells you when it cannot help

Some activations cannot be stabilized across depth by *any* initialization, and AnyInit
says so rather than handing back a network that will not train. The number that decides it
is the Lyapunov slope of the depth map, `χ = d log E[f²] / d log Var[z]`, taken at the
operating point AnyInit initializes to. A relative error in the signal's variance is
multiplied by `χ` at every layer, so whether that matters depends on depth:

| activation | χ | verdict | depth before an error grows 10× |
|---|---|---|---|
| `relu` | 1.000 | marginal | any |
| `relu²` | 2.000 | expansive | 3 |
| `relu³` | 3.000 | expansive | 2 |
| `gelu` | 1.084 | expansive | 28 |
| `silu` | 1.147 | expansive | 16 |
| `elu` | 0.898 | contractive | any |
| `selu` | 0.783 | contractive | any |
| `mish` | 1.052 | expansive | 45 |
| `softplus` | 0.535 | contractive | any |
| `tanh` | 0.359 | infeasible | — |
| `sigmoid` | 0.359 | infeasible | — |

*Contractive* activations damp errors and *marginal* ones hold them at any depth.
*Expansive* ones amplify them: within tenfold for the number of layers in the last column,
and increasingly beyond it, so in the report for a network deeper than that the activation
is called *unstable*. *Infeasible* activations are bounded, so a unit
second moment is out of reach: AnyInit aims at the middle of their variance range instead,
and their `χ` is measured there.

For a positively homogeneous activation χ equals its degree exactly, which is why ReLU
works at arbitrary depth and `relu³` cannot: no scalar per layer holds a map that triples
its own error. The report spells out the consequence:

```
Stability
  relu3   chi= 3.000  sigma*=  0.7148  unstable degree=3
      ! relu3 is homogeneous of degree 3 (chi=3.000), so a relative error grows by
        3.00x per layer and reaches 3.49e+09x over 20 layers. No scalar initialization
        is depth-stable here: reduce depth, insert normalization, or use a degree-one
        activation
```

`report.assert_healthy()` turns that into an exception, which is useful in CI.

## What you get back

```
AnyInit — backend=pytorch, mode=analytic, distribution=normal, center=False, graph=graph
  solve: 6 pass(es), largest objective gap 4.36e-03

  layer                  kind         activation                   fan          scale    pred    meas
  ─────────────────────────────────────────────────────────────────────────────────────────────────────
  conv1                  Conv2d       none                         147       0.082479  1.0000  0.9846
  bn1                    BatchNorm2d  relu→MaxPool2d                 -  gain=0.624688  1.0000  0.9915
  layer1.0.bn2           BatchNorm2d  add→relu                       -  gain=0.058547  1.0032  0.9943
  layer2.0.bn2           BatchNorm2d  add→relu                       -  gain=1.000000  1.0000  1.0001
  layer2.0.downsample.1  BatchNorm2d  add→relu                       -  gain=1.000000  1.0000  1.0001
  layer4.1.bn2           BatchNorm2d  add→relu→AdaptiveAvgPool2d     -  gain=1.413617  1.0000  0.9425
  fc                     Linear       none                         512       0.044194  1.0000  0.9873

Validation
  largest unexplained prediction/measurement gap: 0.00% at bn1  [ok]  (raw gap 0.85%, the rest is sampling noise)
```

The activation column shows the path, not just the name. `add→relu` on both rows means
those two normalizations feed the same post-addition ReLU and were scaled together.
`relu→MaxPool2d` means the target is enforced after the pooling, where the next layer
reads — a 3×3 max pool changes the second moment substantially, so aiming at the ReLU
instead would leave the next block mis-scaled. The same holds for the global average pool
before `fc`. A small gain such as `layer1.0.bn2`'s `0.059` marks a residual block whose
identity path already carries the target on its own.

The validation line is a sanity check, not a pass/fail. It compares the analytic prediction
against one measured forward pass, reports the gap beyond the sampling noise it sits in,
and is never used to pick a scale.

## Framework support

| | PyTorch | Keras 3 | JAX / Flax |
|---|---|---|---|
| traced via | `torch.fx` | `model.operations` | instrumented call on a 16-row probe |
| branches and merges | yes | yes | no — ordered chain |
| activations | modules and functions | layers and fused `activation=` | `jax.nn` calls, and any other recognized from its values |
| transformer layers | `nn.Transformer*` expanded into attention, residuals and FFN | — | — |
| fallback when tracing fails | linear | linear | — |
| weights left untouched (recurrent, bare parameters) | listed in the report | listed in the report | listed in the report |
| parameters | in place | in place | functional, returns a new tree |

JAX keeps parameters outside the model, so:

```python
params, report = anyinit.initialize_params(model, params, (32, 64))
```

The Flax adapter recovers call order rather than graph structure, so a residual addition is
invisible to it and the graph is reported as `linear`. MLPs, sequential CNNs and PINNs are
covered exactly; branched architectures are scaled as if they were sequential.

Because weights are drawn in NumPy from the run's seed and each backend only translates
the layout, **the same architecture and seed give bit-identical weights in all three
frameworks** — which makes cross-framework comparisons reproducible.

## Options

```python
anyinit.initialize(
    model,
    mode="analytic",            # analytic | empirical
    input_spec=(32, 64),        # a shape, a batch, or a callable returning one
    distribution="normal",      # normal | uniform | sinusoidal
    center=False,
    gains={"relu": 2 ** 0.5},   # fix the gain for an activation instead of solving it
    seed=0,
)
```

`mode` chooses where the moments come from: `analytic` propagates the theoretical moment
map through the graph, `empirical` measures real batches.

`center=True` subtracts each output unit's mean weight, so the layer discards its input's
mean rather than propagating it. Off by default; it applies only to layers wide enough to
spare the input direction it costs.

`gains` maps an activation name to a fixed gain. Every layer feeding that activation gets
`gain / √fan_in` (a normalization gets `gain` itself) and is left out of the solve; the
other layers are still solved around it. To see the gain AnyInit would choose:

```python
anyinit.gain("relu")      # 1.4142…, √2
anyinit.gain("silu")      # 1.5588…
anyinit.gain("leaky_relu", negative_slope=0.2)
```

It is the pre-activation standard deviation that lands `E[f(z)²]` on one. For a bounded
activation, where that is out of reach, it is the input scale at the middle of its
reachable variance range. `anyinit.activation_profile(name)` gives the rest of the
picture: `χ`, the moment map, and the stability verdict.

Biases start at zero. The report comes back in either mode; print it, or call
`report.assert_healthy()` to raise on its findings.

## When it helps

Fitting `sin(3x)cos(3y)` with a SiLU MLP, Adam, best loss over 400 steps at each
initialization's own best learning rate:

| depth | PyTorch default | AnyInit |
|---|---|---|
| 10 | 0.00063 | **0.00022** |
| 30 | 0.11610 | **0.00140** |

At depth 30 the stock initialization lets the signal decay to 7.7e-04 by the last layer and
the network barely moves — at two of the three learning rates tried it sits at 0.2467, the
variance of the target, meaning it has learned nothing. At depth 10 the signal survives
either way and the margin is smaller but still three-fold.
`examples/depth_matters.py` reproduces both rows.

The gain is in making the network trainable, not in finding a better optimum, and it grows
with depth because that is when signal propagation becomes the binding constraint.

## Validity

The analytic mode assumes pre-activations are Gaussian, which they are in the
infinite-width limit and approximately at finite width. The approximation degrades for
high-order activations in narrow layers: `E[relu(z)⁶]` measures 2.6× the Gaussian
prediction at width 256 and 1.06× at width 4096. AnyInit detects heavy-tailed signals and
applies a Gamma scale-mixture correction, and the validation pass reports the residual gap.
Rule of thumb: analytic is reliable for `χ ≲ 1.2` at width ≥ 256; outside that, use
`empirical`, which measures instead of assuming.

The analytic mode predicts the ensemble, while a model holds one draw, so a layer's
realized statistics drift from the prediction and the drift accumulates with depth. For a
20-layer ReLU MLP the final second moment (target 1) lands, across 16 seeds:

| width | analytic | analytic, `center=True` | empirical |
|---|---|---|---|
| 128 | 0.11 – 2.25 | 0.78 – 1.22 | 1.00 |
| 512 | 0.58 – 1.51 | 0.98 – 1.03 | 1.00 |

`center=True` removes the term responsible, at the cost of one input direction per layer;
the empirical mode corrects the particular draw. Reproduce with
`python docs/experiments/draw_drift.py`.

Merges assume branch independence. In a residual block the branch and the skip share an
input, so that is not strictly true; the error is modest and `empirical` does not rely on
it.

`sinusoidal` has correlated rows, which the analytic recursion only approximates — a
structured row resonates with structure in the activations where a random one averages it
out. It is flagged in the report; use `empirical` with it, which holds the second moment to
a few percent for every distribution.

## Development

```bash
pip install -e '.[dev]' torch tensorflow jax flax
pytest                                  # backend tests skip when a framework is absent
ruff check src tests examples docs
ruff format --check src tests examples docs
mypy src                                # strict

pip install -r docs/requirements.txt
mkdocs serve                            # the documentation site, locally
```

The architecture is a framework-free core plus one adapter per framework:

```
anyinit/core/       quadrature, profiles, stability, graph IR, topology, solvers
anyinit/backends/   pytorch.py, keras.py, jax.py
```

`anyinit/core` imports no framework, and a test walks its AST to keep it that way. Adding a
backend means implementing `Backend` — trace to the IR, convert weight layouts, and report
moments from a forward pass. None of the numerics is touched.

[docs/design.md](https://github.com/jmiravet/AnyInit/blob/main/docs/design.md) covers the architecture and what implementing a backend
involves. [docs/reference.md](https://github.com/jmiravet/AnyInit/blob/main/docs/reference.md) has the activation moment table and a
glossary. [docs/experiments/](https://github.com/jmiravet/AnyInit/tree/main/docs/experiments/) reproduces the measurements quoted here.

## References

Variance scaling comes from Glorot & Bengio (2010) and He et al. (2015), whose rectifier
gain falls out of the moment map as the degree-one case. Treating propagation as a
dynamical system, and the order parameter the χ diagnostic reports, come from Poole et al.,
*Exponential expressivity through transient chaos* (2016) and Schoenholz et al., *Deep
Information Propagation* (2017). The empirical mode generalizes LSUV (Mishkin & Matas,
2016) from a sequence to a graph. Fixed points of the variance map, and the SELU constants
the reference table reproduces, are from Klambauer et al. (2017). The `sinusoidal`
distribution is Fernandez-Hernandez et al. (2025).

Evaluating the moment map numerically, rather than deriving it per activation, is what
makes an unseen activation a first-class input. Full list in
[docs/reference.md](https://github.com/jmiravet/AnyInit/blob/main/docs/reference.md).

## License

MIT
