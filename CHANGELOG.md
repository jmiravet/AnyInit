# Changelog

## Unreleased

### Fixed

- An embedding table tied to the output layer was scaled as a lookup alone, which leaves
  the logits at standard deviation √d_model and an initial loss that grows with width. A
  PyTorch model with a `Linear` head escaped only because the head was written last, and
  the report said nothing either way. Tied tables are now detected (a shared `Parameter`,
  or `F.linear` and `@` on an embedding's weight, in PyTorch; `Embed.attend` in Flax;
  `reverse=True` calls of a tied `ReversibleEmbedding` in Keras), given the output layer's
  scale, held at it through the solve, and reported. The docs give the measurements behind
  the choice and what to do beyond initialization: a √d multiplier on the lookup, and the
  table's learning rate.
- Keras: a subclass of `Embedding` was laid out as a dense kernel.

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
