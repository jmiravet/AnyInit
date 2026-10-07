# Why a gain table is not enough

The usual recipe picks one constant per activation — `√2` for ReLU, `5/3` for tanh — and
scales each layer by `gain / √fan_in`. That is exact only for ReLU-like activations. For
everything else the right gain depends on the variance actually arriving, and the error
compounds with depth.

AnyInit computes, for any activation `f`, the moments of `f(z)` when `z` is Gaussian, and
solves for one scale per layer from that.

## The signal through a deep MLP

`E[a²]` after each activation of a 20-layer, 256-wide MLP, mean ± standard deviation over
five seeds (target 1):

| activation | gain table, layer 1 | layer 10 | layer 20 | AnyInit analytic, layer 20 | AnyInit empirical, layer 20 |
|---|---|---|---|---|---|
| ReLU | 1.00 ± 0.01 | 1.21 ± 0.58 | 1.20 ± 0.33 | 0.81 ± 0.35 | 1.00 ± 0.00 |
| Tanh | 0.58 ± 0.00 | 0.42 ± 0.00 | 0.42 ± 0.00 | 0.50 ± 0.00 \* | 0.50 ± 0.00 \* |
| Sigmoid | 0.29 ± 0.00 | 0.27 ± 0.01 | 0.27 ± 0.01 | 0.38 ± 0.01 \* | 0.38 ± 0.01 \* |
| SiLU | 0.80 ± 0.01 | 0.019 ± 0.009 | **2.0e-05** | 3.2 ± 1.9 † | 1.02 ± 0.03 |

ReLU is the case the table was derived for, and it holds. SiLU borrows ReLU's gain, as is
common practice, and the signal vanishes.

\* Bounded activations cannot reach 1; AnyInit targets the middle of what they can reach.

† SiLU is [expansive](guide/stability.md) (`χ = 1.147`), so over 20 layers one draw
strays from the analytic prediction. The report says so; `empirical` corrects it.

Reproduce with `python docs/experiments/gain_table.py`.

## Effect on training

Fitting `sin(3x)cos(3y)` with a SiLU MLP and Adam, best loss over 400 steps at each
initialization's own best learning rate:

| depth | PyTorch default | AnyInit |
|---|---|---|
| 10 | 0.00063 | **0.00022** |
| 30 | 0.11610 | **0.00140** |

At depth 30 the default initialization lets the signal decay to 7.7e-04 by the last layer
and the network barely learns. The benefit is in making deep networks trainable, so it
grows with depth. See the [tutorial](tutorials/depth.md).
