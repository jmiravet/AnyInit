# Reproducing the measurements

The figures quoted in the README and in `../reference.md` come from these.  They are kept
out of the test suite because they take minutes rather than seconds.

| script | what it measures |
|---|---|
| `quadrature_accuracy.py` | Gauss-Legendre against Gauss-Hermite and Monte Carlo on `relu³`, whose moments are known exactly |
| `gaussian_breakdown.py` | How far a high moment departs from the Gaussian prediction with width, and whether the Gamma correction tracks it |
| `stability_table.py` | `χ` and `σ*` for every builtin activation and for `relu^p` |
| `gain_table.py` | `E[a²]` through a 20-layer MLP under the usual gain table and under both AnyInit modes |
| `draw_drift.py` | How far one draw of a deep ReLU MLP lands from the analytic prediction, with and without `center=True`, against the empirical mode |
| `depth_sweep.py` | Measured rescaling across depth and width for `relu^p`, showing which degrees survive |

Run any of them directly:

```bash
python docs/experiments/stability_table.py
```

`depth_sweep.py`, `draw_drift.py` and `gain_table.py` need PyTorch; the others need only NumPy.
