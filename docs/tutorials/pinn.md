# A PINN with relu²

Solves `u'' = f` on `[0, 1]` with `u(0) = u(1) = 0`, imposing the boundary condition by
construction. The activation is `relu(x)²`, homogeneous of degree two, which AnyInit
reports as `χ = 2`: not depth-stable, usable at the three layers this network has.

What AnyInit contributes here is the diagnosis rather than a training win: it says up
front that `relu(x)²` cannot survive depth. The training numbers it prints are not
evidence either way, because a loss on a second derivative is badly conditioned and a
single-seed comparison measures the learning rate more than the initialization; the
example's docstring gives the details.

```bash
python examples/pinn_relu2.py
```

```python
--8<-- "examples/pinn_relu2.py"
```
