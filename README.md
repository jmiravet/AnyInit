# AnyInit

[![CI](https://github.com/jmiravet/AnyInit/actions/workflows/ci.yml/badge.svg)](https://github.com/jmiravet/AnyInit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jmiravet/AnyInit/graph/badge.svg)](https://codecov.io/gh/jmiravet/AnyInit)
[![PyPI](https://img.shields.io/pypi/v/anyinit)](https://pypi.org/project/anyinit/)
[![Python](https://img.shields.io/pypi/pyversions/anyinit)](https://pypi.org/project/anyinit/)
[![Docs](https://img.shields.io/badge/docs-jmiravet.github.io%2FAnyInit-blue)](https://jmiravet.github.io/AnyInit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/jmiravet/AnyInit/blob/main/LICENSE)

Initialize any model, in any framework, correctly — with one call.

```python
import anyinit

report = anyinit.initialize(model)
print(report)
```

`model` can be a PyTorch module, a Keras model or a Flax module. AnyInit traces it, finds
which activation follows each layer, and scales every weight so the signal neither dies
nor explodes on its way through. Nothing to configure, nothing to look up.

## Install

```bash
pip install anyinit
```

AnyInit depends on NumPy alone and uses whichever framework your model comes from.

## What "any" means

**Any framework.** PyTorch, Keras 3 (on TensorFlow, JAX or PyTorch) and Flax, detected
from the model. The same seed gives the same weights in all three.

**Any activation.** AnyInit measures the function itself instead of looking it up in a
table, so a new activation is one decorator away:

```python
@anyinit.register_activation
class ReLUCubed(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) ** 3
```

**Any architecture.** Branches, residual additions, concatenations, normalization,
pooling, dropout, attention and embeddings.

**Theory or measurement.** Scale from theory, with no data, or from real batches:

```python
anyinit.initialize(model, "analytic", (32, 3, 224, 224))   # shapes only, milliseconds
anyinit.initialize(model, "empirical", batch)              # measures, assumes nothing
```

## It tells you when it cannot help

Some activations cannot be kept stable across depth by any initialization. AnyInit says
so in the report instead of handing back a network that will not train:

```
Stability
  relu3   chi= 3.000  sigma*=  0.7148  unstable degree=3
      ! relu3 is homogeneous of degree 3 (chi=3.000), so a relative error grows by
        3.00x per layer and reaches 3.49e+09x over 20 layers. No scalar initialization
        is depth-stable here: reduce depth, insert normalization, or use a degree-one
        activation
```

Call `report.assert_healthy()` to turn that into an exception, for example in CI.

## Learn more

The [documentation](https://jmiravet.github.io/AnyInit/) covers:

- [Reading the report](https://jmiravet.github.io/AnyInit/guide/report/)
- [Analytic or empirical](https://jmiravet.github.io/AnyInit/guide/modes/)
- [Stability across depth](https://jmiravet.github.io/AnyInit/guide/stability/)
- [Frameworks](https://jmiravet.github.io/AnyInit/guide/frameworks/)
- [Options](https://jmiravet.github.io/AnyInit/guide/options/)
- [Why a gain table is not enough](https://jmiravet.github.io/AnyInit/why/), with measurements
- [Tutorials](https://jmiravet.github.io/AnyInit/tutorials/custom-activation/) and the
  [API](https://jmiravet.github.io/AnyInit/api/)

Contributions are welcome; see
[Contributing](https://jmiravet.github.io/AnyInit/contributing/).

## References

Variance scaling comes from Glorot and Bengio (2010) and He et al. (2015), whose rectifier
gain falls out of the moment map as the degree-one case. Treating signal propagation as a
dynamical system, and the order parameter behind the `χ` diagnostic, come from Poole et al.
(2016) and Schoenholz et al. (2017). The empirical mode generalizes LSUV (Mishkin & Matas,
2016) from a sequence of layers to a graph. Fixed points of the variance map, and the SELU
constants the reference table reproduces, are from Klambauer et al. (2017). The
`sinusoidal` distribution is from Fernández-Hernández et al. (2025).

What is new is evaluating the moment map numerically, by Gaussian quadrature, instead of
deriving it per activation, which is why AnyInit accepts an activation it has never seen.

Fernández-Hernández, A., Mestre, J. I., Dolz, M. F., Duato, J., & Quintana-Ortí, E. S.
(2025). Sinusoidal initialization, time for a new start. In *Advances in Neural
Information Processing Systems* (Vol. 38). https://doi.org/10.48550/arXiv.2505.12909

Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward
neural networks. In *Proceedings of the Thirteenth International Conference on Artificial
Intelligence and Statistics* (pp. 249–256). PMLR.
https://proceedings.mlr.press/v9/glorot10a.html

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing
human-level performance on ImageNet classification. In *Proceedings of the IEEE
International Conference on Computer Vision* (pp. 1026–1034). IEEE.
https://doi.org/10.1109/ICCV.2015.123

Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S. (2017). Self-normalizing neural
networks. In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 971–980).
https://arxiv.org/abs/1706.02515

Mishkin, D., & Matas, J. (2016). All you need is a good init. In *International Conference
on Learning Representations*. https://arxiv.org/abs/1511.06422

Poole, B., Lahiri, S., Raghu, M., Sohl-Dickstein, J., & Ganguli, S. (2016). Exponential
expressivity in deep neural networks through transient chaos. In *Advances in Neural
Information Processing Systems* (Vol. 29, pp. 3360–3368).
https://arxiv.org/abs/1606.05340

Schoenholz, S. S., Gilmer, J., Ganguli, S., & Sohl-Dickstein, J. (2017). Deep information
propagation. In *International Conference on Learning Representations*.
https://arxiv.org/abs/1611.01232
