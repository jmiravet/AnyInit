# Analytic or empirical

```python
anyinit.initialize(model, "analytic", (32, 3, 224, 224))  # no data, no forward pass
anyinit.initialize(model, "empirical", batch)  # no assumptions
```

**`analytic`** (the default) propagates moments through the graph without running the
model; `input_spec` contributes shapes and nothing else. It is deterministic, data-free,
and takes milliseconds.

**`empirical`** pushes real batches through, measures what arrives, and corrects layer by
layer in topological order, so each layer is exact given everything upstream.

Both share the graph, the layer adapters and the target, so switching changes the method,
not the goal. Start with `analytic`; reach for `empirical` when the report warns about the
analytic assumptions.

## When the analytic mode is less accurate

**Narrow layers with high-order activations.** The analytic mode assumes pre-activations
are Gaussian, which holds well at large width. `E[relu(z)⁶]` measures 2.6× the Gaussian
prediction at width 256, but only 1.06× at width 4096. AnyInit detects heavy tails and
corrects for them, and the validation reports what remains.

Rule of thumb: analytic is reliable for `χ ≲ 1.2` at width ≥ 256.

**One draw against the ensemble.** The analytic mode predicts the average over random
draws, and a model holds a single draw. For a 20-layer ReLU MLP, the final second moment
(target 1) lands, across 16 seeds:

| width | analytic | analytic, `center=True` | empirical |
|---|---|---|---|
| 128 | 0.11 – 2.25 | 0.78 – 1.22 | 1.00 |
| 512 | 0.58 – 1.51 | 0.98 – 1.03 | 1.00 |

[`center=True`](options.md#center) removes most of the spread; `empirical` removes all of
it. Reproduce with `python docs/experiments/draw_drift.py`.

**Residual blocks.** Merges assume the branches are independent, which a branch and its
skip connection are not quite. The error is modest, and `empirical` does not rely on it.

**`distribution="sinusoidal"`.** Its rows are correlated, which the analytic recursion only
approximates. The report flags it; use `empirical`.
