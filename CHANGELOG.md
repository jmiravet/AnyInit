# Changelog

## 0.3.0

### Tied embeddings and constant factors

An embedding table tied to the output layer was scaled as a lookup only. That leaves the
logits at standard deviation √d_model, so the initial loss grows with width. This release
detects tied tables, gives them the output layer's scale and keeps that scale fixed
during the solve. The report says what the lookup ends up with.

Along the way, the analytic mode now reads constant factors (`x * c`, `x / c`,
`Rescaling`, …) instead of treating them as the identity.

### Changes

**Tied tables** (`core/tying.py`)

- Backends mark every node that reads a tied table with a shared `meta["tied"]` key:
  - PyTorch: a shared `Parameter`, or `F.linear` / `@` on an embedding's weight
  - Flax: `Embed.attend`
  - Keras: `reverse=True` calls of a tied `ReversibleEmbedding`
- `tying.find` groups those nodes, gives the table the output role's scale
  `1/(c·√d_model)` and passes it to the solve as fixed. Here `c` is any constant factor
  the model applies around the output layer.
- The report gets one line per tied table, linking to the new experiment page.

**Constant factors** (new `NodeKind.SCALE`)

- PyTorch: `x * c`, `c * x`, `x / c`, and scalar or per-channel buffers and parameters
- Keras: `x * c`, `x / c` and `Rescaling`
- Flax: any scalar factor between two layers, read off the probe
- `transfer.through_scale` propagates moments through these nodes, and `SCALE` is
  transparent to the layer/activation pairing.

**Keras fixes**

- Operations holding a constant, such as `x * 2.0` in a functional model, can now be
  replayed. Before, validation and the empirical mode measured nothing after them.
- Subclasses of `Embedding` are no longer laid out as dense kernels.

**Docs**

- New experiment page, `docs/experiments/tied-embeddings.md`, with the measurements
  behind the choice and what to do beyond initialization (a √d multiplier on the lookup,
  and the table's learning rate).

### Behavior change

The analytic mode now gives different results for models that multiply by a constant.
Those results now match the empirical mode, which measured the real network all along.
One thing to note: the solver now compensates for a factor the user added on purpose. For
example, in `h + 0.5 * relu(f(h))` the weights of `f` double. The empirical mode already
did this.

| Model | Weight | before | after | empirical |
|---|---|---|---|---|
| `h + 0.5 * relu(f(h))` | `f.weight` | 0.177 | 0.354 | 0.345 |
| LayerScale `h + γ·f(h)` | `f3.weight` | 0.125 | 0.177 | 0.184 |

### Testing

- Full suite: 419 passed, none skipped (PyTorch, Keras on torch, Flax). The 382 existing
  tests are unchanged, and there are 37 new ones.
- `ruff check`, `ruff format --check` and `mypy` are clean.
- 0.2.0 and this release were run on the same set of ordinary models, with the same seed,
  in both modes, comparing the std of every weight and the report warnings. Results are
  identical everywhere except the two constant-factor rows above.
  - PyTorch: MLPs, CNN with BatchNorm, ResNet18, MobileNetV3, EfficientNet-B0,
    ConvNeXt-tiny, ViT, `nn.TransformerEncoder`, an untied LM, and the torch.fx fallback.
  - Models that share weights without tying: shared `Linear` weights, an embedding looked
    up twice, `h @ linear.weight.T`, GLU-style gating.
  - Keras and Flax: MLPs, CNNs, residuals, gating, LMs, shared layers.

## 0.2.0

### Fixed

- Under Keras and JAX the synthesized validation batch replayed the random stream of the
  first layer's weights, which biased the measured second moments (1.25 instead of 1.0
  after the first ReLU). Input batches now draw from a stream of their own.
- PyTorch models on CUDA, or in a dtype other than float32, skipped validation because the
  synthesized batch was created on the CPU in float32. It now follows the model.
- `assert_healthy()` passed a report whose requested validation had been skipped; it now
  fails, citing the reason.
- `assert_healthy()` flagged correctly initialized deep networks: the sampling noise
  allowed for did not grow with depth, and 98% of seeds of a 30-layer ReLU MLP failed.
  Each layer's noise floor now accumulates the draw noise of the layers above it, back to
  the last normalization, and is symmetric in log scale.
- Initializing a PyTorch model in a process that had imported TensorFlow first crashed
  with a segmentation fault: shape inference used fake tensors, which load Triton. Shapes
  now come from a one-row probe, run only for models with adaptive pooling.
- JAX: an activation called inside another one (a registered function built on
  `jnp.tanh`, say) was recorded as a second activation, or in its place. Nested calls are
  now one step, and the activations seen being called are checked against the values
  that reach the next layer; a mismatch is identified among the known and registered
  activations, including registered functions defined inside another function.
- JAX: an activation stored as a module field was taken for no activation when the
  incoming weights were small (`normal(0.02)`), since the probe's signal fell below an
  absolute tolerance, and on the last layer when the model returned several arrays.
  The probe now runs with unit-scale weights, values are compared relative to their
  magnitude, and the last layer is matched against every leaf of the output.
- Duplicate registrations and natively registered activations profiled without a
  backend raised a bare `ValueError`; they now raise `ConfigError`, which subclasses it.
  `gain()` also finds the backend of a registered function that reaches its framework
  through a closure.
- The moment table in the reference docs failed to match the code on macOS: a mean that
  is zero in exact arithmetic comes out of the quadrature as ±1e-17, its sign set by the
  platform's libm, and printed as `-0.0000000`. Zeros are now printed without a sign.

## 0.1.0

First release.

### Added

- `initialize(model, mode, input_spec, ...)` — one call initializes any model, with
  the framework detected from it. The remaining options are explicit keywords:
  `distribution`, `center`, `gains`, `seed` and `params`. PyTorch, Keras 3 (on any of
  its backends) and JAX (via Flax), each an optional dependency, so installing one never
  imports another.
- `initialize_params(model, params, ...)` for functional frameworks, which return a new
  parameter tree rather than mutating one.
- Two solve modes. `analytic` propagates moments through the graph and never runs the
  model; `empirical` measures real batches and assumes nothing distributional. They share
  the graph, the layer adapters and the target policy.
- `register_activation` for arbitrary activations. Their moment maps are computed by
  Gauss–Legendre quadrature over the Gaussian, exact to machine precision across kinks.
- `anyinit.gain(name, **params)`: the gain AnyInit gives an activation, `√2` for ReLU.
  `initialize(gains={...})` fixes the gain for an activation instead of solving for it.
- Depth-stability diagnosis. Every activation gets `χ = d log E[f²] / d log Var[z]`, which
  equals the homogeneity degree for homogeneous activations and so identifies the ones no
  scalar initialization can stabilize.
- `InitReport`: the scale chosen for every layer, the graph fidelity actually achieved, a
  measured validation pass sized against its own sampling noise, and anything AnyInit
  could not do. `assert_healthy()` raises on the findings.
- Weights are drawn in NumPy in a canonical layout from the run's seed, so the same
  architecture and seed produce bit-identical weights under all three frameworks.
