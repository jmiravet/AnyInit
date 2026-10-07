# Stability across depth

Some activations cannot be held stable across depth by *any* initialization. AnyInit
detects this and says so in the report.

## The number that decides it

A small relative error in the signal's variance is multiplied by `χ` at every layer:

- `χ < 1`: errors shrink, any depth works.
- `χ ≈ 1`: errors hold steady, as with ReLU.
- `χ > 1`: errors grow, so depth is limited.

(Formally, `χ = d log E[f²] / d log Var[z]`, taken at the point AnyInit initializes to.)

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

For `relu(x)ⁿ`, `χ` is exactly `n`. That is why ReLU works at any depth and `relu³` does
not: no single scale per layer can hold a map that triples its own error.

## Verdicts

| verdict | condition |
|---|---|
| contractive | `χ < 0.98`: errors shrink with depth |
| marginal | `0.98 ≤ χ ≤ 1.02`: errors neither grow nor shrink |
| expansive | `χ > 1.02`, but `χ^depth ≤ 10`: errors grow, at most tenfold over the network |
| unstable | `χ^depth > 10`: no scalar initialization holds the signal at this depth |
| infeasible | the activation cannot reach the target second moment at all |

An activation alone is at most *expansive*; it becomes *unstable* in a network deeper than
the last column of the table above.

Bounded activations (`tanh`, `sigmoid`) cannot reach a unit second moment, so AnyInit aims
at the middle of the range they can reach, and says so.

## In the report

```
Stability
  relu3   chi= 3.000  sigma*=  0.7148  unstable degree=3
      ! relu3 is homogeneous of degree 3 (chi=3.000), so a relative error grows by
        3.00x per layer and reaches 3.49e+09x over 20 layers. No scalar initialization
        is depth-stable here: reduce depth, insert normalization, or use a degree-one
        activation
```

`anyinit.activation_profile(name)` gives `χ` and the verdict for any activation without
building a model. Regenerate the table with `python docs/experiments/stability_table.py`.
