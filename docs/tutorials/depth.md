# When initialization decides the outcome

Fits `sin(3x)cos(3y)` with a SiLU MLP at two depths, with and without AnyInit, and prints
the activation levels next to the losses so the mechanism is visible rather than asserted.
Initialization is decisive when signal propagation is the binding constraint, and the
margin shrinks as that stops being true.

```bash
python examples/depth_matters.py
```

```python
--8<-- "examples/depth_matters.py"
```
