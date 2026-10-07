# Contributing

## Setup

```bash
pip install -e '.[dev]' torch tensorflow jax flax
pytest                                  # backend tests skip when a framework is absent
ruff check src tests examples docs
ruff format --check src tests examples docs
mypy src                                # strict

pip install -r docs/requirements.txt
mkdocs serve                            # the documentation site, locally
```

[Design](design.md) explains how the code is laid out.

## Adding a backend

Implement `anyinit.backends.base.Backend` and add it to `_BACKENDS` in
`anyinit/backends/__init__.py`. None of the numerics is touched. The contract is:

- `handles(model)` — recognize the model from its class hierarchy, without importing the
  framework. `module_roots` does the inspection.
- `build_graph(model, input_spec)` — trace into a `ModelGraph`.
- `eval_elementwise(fn, x)` — apply a native callable to quadrature abscissas.
- `read_weight` / `write_weight` / `write_bias` / `write_gain` — canonical layout in and
  out. `write_gain` assigns rather than multiplies, so `initialize` stays idempotent.
- `forward_taps(model, inputs, taps, training=True)` — one forward pass, returning moments
  at the named nodes. `training` must normally be true: at initialization a batch-norm
  layer's running statistics are still (0, 1), so in inference mode it scales by gamma
  instead of normalizing. Running statistics must be restored afterward, and random state
  fixed during the pass, so that dropout draws the same masks every time and the empirical
  solver sees a deterministic function of the scales.
- `unscaled_weights(model, graph)` — every weight no graph node writes, with the reason.
- `label` — how the report names the backend; Keras adds what it runs on, `keras[jax]`.
- `finalize(model)` — new parameters for a functional framework, `None` otherwise.

`tests/test_cross_backend.py` then checks that the new backend produces the same weights
as the others from the same seed.
