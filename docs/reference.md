# Reference

## Activation moments

Exact for `x ~ N(0, 1)`, verified by quadrature. `gain` is the second-moment gain
`1/√E[f²]`, `σ*` the pre-activation standard deviation at which `E[f²] = 1`, and `χ` the
Lyapunov slope of the depth map at `σ*`. For `tanh` and `sigmoid`, which never reach
`E[f²] = 1`, `χ` is the slope of the output variance at the point AnyInit substitutes, the
middle of their reachable variance range.

| `f` | `E[f]` | `E[f²]` | gain | `χ` | `σ*` | degree |
|---|---|---|---|---|---|---|
| `relu` | 0.3989423 | 0.5000000 | 1.4142136 | 1.000 | 1.4142 | 1 |
| `relu²` | 0.5000000 | 1.5000000 | 0.8164966 | 2.000 | 0.9036 | 2 |
| `relu³` | 0.7978846 | 7.5000000 | 0.3651484 | 3.000 | 0.7148 | 3 |
| `gelu` | 0.2820948 | 0.4252215 | 1.5335304 | 1.084 | 1.4680 | — |
| `silu` | 0.2066210 | 0.3557755 | 1.6765325 | 1.147 | 1.5588 | — |
| `elu` | 0.1605206 | 0.6449454 | 1.2451983 | 0.898 | 1.2780 | — |
| `selu` | 0.0000000 | 1.0000000 | 1.0000000 | 0.783 | 1.0000 | — |
| `mish` | 0.2404039 | 0.4523422 | 1.4868476 | 1.052 | 1.4515 | — |
| `softplus` | 0.8060592 | 0.9212459 | 1.0418668 | 0.535 | 1.0831 | — |
| `tanh` | 0.0000000 | 0.3942945 | 1.5925374 | 0.359 | unreachable | — |
| `sigmoid` | 0.5000000 | 0.2933790 | 1.8462285 | 0.359 | unreachable | — |

Closed forms for the rectifier powers: `E[relu(x)ⁿ] = ½E[|x|ⁿ]`, so
`E[relu³] = √(2/π)` and `E[relu³²] = ½E[x⁶] = 7.5`.

`selu` reaching `σ* = 1` exactly is its design property. Regenerate the table with
`python docs/experiments/stability_table.py`.

The meaning of `χ` and the stability verdicts are in
[Stability across depth](guide/stability.md).

## Glossary

| Term | Meaning |
|---|---|
| **Activation profile** | The map `M_f: (μ, σ²) → (E[a], E[a²], E[a⁴])` |
| **χ (chi)** | Lyapunov slope of the depth map, `d log E[f²] / d log Var[z]`. One is the edge of chaos; above one, a variance error is amplified per layer |
| **σ\*** | Pre-activation standard deviation at which `E[f²]` meets its target |
| **Canonical layout** | `(fan_out, fan_in, *receptive_field)`, the order the core reasons in |
| **Dominating ancestor** | A scalable node upstream of an activation, reachable without crossing another activation |
| **Measurement node** | Where an activation's target is enforced: the end of the pooling, dropout and reshapes that follow it |
| **Fidelity** | `graph` when the full DAG was observed, `linear` when only an ordered chain was |
| **Scale mixture** | `z │ S ~ N(0, S)` with `S` random; the model of a pre-activation whose input has heavy tails |
| **Objective** | The condition one activation is driven to, as a metric and a value |

## Further reading

The papers AnyInit builds on, and what each contributes, are listed under
[References](index.md#references) on the home page.
