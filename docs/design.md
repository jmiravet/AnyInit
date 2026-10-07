# Design

Notes for anyone reading or extending the code.

## Two layers

```
┌──────────────────────────────────────────────────────┐
│ anyinit.core — imports no framework                 │
│ quadrature · profiles · stability · graph IR ·       │
│ topology · transfer · fan · distributions · solvers  │
└───────────────────────┬──────────────────────────────┘
                        │  ModelGraph + numpy.ndarray
┌───────────────────────┴──────────────────────────────┐
│ anyinit.backends — one module per framework         │
│ pytorch.py · keras.py · jax.py                       │
│ tracing · weight layouts · forward passes · taps     │
└──────────────────────────────────────────────────────┘
```

The core never holds a tensor. It speaks `ModelGraph` — an IR of typed nodes — and
`numpy.ndarray` in a canonical layout. Backends trace the native model into that IR,
translate layouts, read and write parameters, and run forward passes returning
*statistics* rather than tensors.

## Principles

1. **The core imports no framework.** `tests/test_core_isolation.py` walks the AST of
   every core module to enforce it, and runs a subprocess to check that importing
   `anyinit` loads none of them.
2. **Lazy imports.** A framework is imported inside the method that needs it, never at
   module scope, so installing one never pulls in another.
3. **Reinitialize, not rescale.** The analytic recursion assumes `W` is i.i.d. zero-mean
   with variance `σ_w²`; drawing the weights makes that true by construction.
4. **One graph, two solvers.** `analytic` and `empirical` share the IR, the topology, the
   layer adapters, the target policy and the report. They differ only in where the
   moments come from.
5. **Mode purity.** No measurement ever sets an `analytic` scale. The model is run only
   for the validation pass, which runs when `input_spec` is given and whose result
   reaches the report alone, and, under Flax, for a 16-row probe that identifies
   activations (below). Shapes come from fake tensors,
   which carry no data. `empirical` never consults a Gaussian profile to set a scale.
6. **Weights drawn in NumPy.** The same seed gives the same weights under every backend.
7. **Fail visibly.** Anything AnyInit cannot do correctly goes into the report rather
   than being approximated silently.

## Canonical weight layout

The core reasons in `(fan_out, fan_in, *receptive_field)`. Each backend translates:

| Layer | Canonical | PyTorch | Keras | Flax |
|---|---|---|---|---|
| Dense | `(out, in)` | `(out, in)` | `(in, out)` | `(in, out)` |
| Conv2D | `(out, in/g, kh, kw)` | `(out, in/g, kh, kw)` | `(kh, kw, in, out)` | `(kh, kw, in, out)` |
| ConvTranspose2D | `(out, in/g, kh, kw)` | `(in, out/g, kh, kw)` | `(kh, kw, out, in)` | `(kh, kw, in, out)` |
| Embedding | `(num, dim)` | `(num, dim)` | `(num, dim)` | `(num, dim)` |

A transposed convolution stores its input and output axes the other way round from an
ordinary one, so fans computed on the native shape come out inverted.

## Tracing

| Framework | Mechanism | Fidelity |
|---|---|---|
| PyTorch | `torch.fx` symbolic trace; shapes from fake-tensor propagation | graph; falls back to a linear chain from `named_modules` when tracing fails |
| Keras | `model.operations` and each operation's inbound nodes | graph; linear for an unwired Sequential |
| Flax | Flax layers and `jax.nn` activations wrapped inside a context manager, then the model applied to a 16-row probe | linear |

PyTorch's `nn.Transformer*` layers are traced as single modules, so the adapter expands
each into its real structure — attention, residual additions, normalizations and the
feed-forward block — with both `norm_first` layouts. The expanded nodes have no FX node of
their own and are measured through hooks on the submodules that produce or consume them.
Adaptive pooling records no window; the fake-tensor shapes supply it.

A jaxpr is too far from the source to read activations off: most `jax.nn` functions lower
to opaque wrappers and a bias add is indistinguishable from a residual one. The
instrumented trace recovers call order instead, and each layer reports `scope.path`, which
is its key in the parameter tree. That yields an ordered chain rather than a DAG, since
nothing intercepts `+`, and the graph is marked `linear`. An activation that never passes
through `jax.nn` — a module field, a closure, a lambda — is not seen being called, so the
probe's values identify it: wherever one layer's output reaches the next transformed, the
transform is matched against the known and registered activations.

Whatever a backend reaches but cannot scale — recurrent weights, attention layers it has no
adapter for, bare parameters — is listed in the report with the reason, never skipped
silently.

## Adding a backend

Implement `anyinit.backends.base.Backend` and add it to `_BACKENDS` in
`anyinit/backends/__init__.py`. The contract is:

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

None of the numerics is touched. `tests/test_cross_backend.py` then checks that the new
backend produces the same weights as the others from the same seed.

## Where things live

| Concern | Module |
|---|---|
| Gaussian integration, kink detection | `core/quadrature.py` |
| NumPy reference activations | `core/activations.py` |
| Moment map of an activation | `core/profile.py` |
| Activation registry | `core/registry.py` |
| `χ`, feasibility, verdicts | `core/stability.py` |
| Typed graph IR | `core/graph.py` |
| Layer/activation pairing | `core/topology.py` |
| Moment transfer per node kind | `core/transfer.py` |
| Canonical shapes and fan arithmetic | `core/fan.py` |
| Weight samplers | `core/distributions.py` |
| Solvers | `core/analytic.py`, `core/empirical.py` |
| Orchestration | `_run.py` |
