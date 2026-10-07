# Same weights in every framework

Weights are drawn in NumPy from the run's seed in a canonical layout, and each backend only
translates that layout into its own. The same architecture and seed therefore produce the
same numbers in PyTorch, Keras and Flax. That makes cross-framework comparisons
reproducible, and checks that nothing framework-specific leaked into the solver.

Flax parameters are immutable, so Flax goes through `initialize_params`, which returns a
new parameter tree. The example skips any framework that is not installed.

```bash
python examples/multibackend.py
```

```python
--8<-- "examples/multibackend.py"
```
