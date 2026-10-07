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

```bash
python examples/custom_activation.py
```

```python
--8<-- "examples/custom_activation.py"
```
