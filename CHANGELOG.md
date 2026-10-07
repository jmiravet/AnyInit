# Changelog

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
