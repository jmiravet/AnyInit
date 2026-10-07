# A new activation

AnyInit has no table of activations to fall back on: every activation, built in or not, is
described by its moment map, computed by Gaussian quadrature. So an activation it has
never seen needs one line to register, after which both modes use it.

The example registers `relu(x)³` as an `nn.Module` subclass. Registering a class rather
than a function keeps it a single node in the traced graph, instead of `torch.fx`
tracing into it as a `relu` and a `pow`. It then prints the profile AnyInit computed, checked
against the closed form, and initializes a 2-layer and a 20-layer network. The stability
diagnosis reports `χ = 3`: a degree-three activation amplifies any variance error threefold per layer, which
no scalar initialization can correct.

Written against NumPy, an activation works with every backend without importing any
framework:

```python
anyinit.register_activation(lambda x: np.maximum(x, 0) ** 3, name="relu3")
```

The moment map is computed by Gauss–Legendre quadrature in about 128 evaluations, exact to
machine precision even across the kink. For `relu³` it recovers `E[f] = 0.7978846` and
`E[f²] = 7.5` to twelve digits, where 10⁷ Monte Carlo samples still carry ~1e-3 error.

```bash
python examples/custom_activation.py
```

```python
--8<-- "examples/custom_activation.py"
```
