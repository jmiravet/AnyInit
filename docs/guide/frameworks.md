# Frameworks

AnyInit detects the framework from the model, and never imports the others.

```python
anyinit.available_backends()  # installed here, e.g. ('pytorch',)
anyinit.known_backends()  # ('keras', 'jax', 'pytorch')
```

## JAX / Flax

Flax keeps parameters outside the model, so it has its own call, which returns a new
parameter tree:

```python
params, report = anyinit.initialize_params(model, params, (32, 64))
```

The Flax adapter sees the order of calls, not the full graph, so a residual addition is
invisible to it. MLPs, sequential CNNs and PINNs are handled exactly; branched
architectures are scaled as if they were sequential, and the report says `graph=linear`.

## Support

| | PyTorch | Keras 3 | JAX / Flax |
|---|---|---|---|
| traced via | `torch.fx` | `model.operations` | instrumented call on a 16-row probe |
| branches and merges | yes | yes | no — ordered chain |
| activations | modules and functions | layers and fused `activation=` | `jax.nn` calls, and any other recognized from its values |
| transformer layers | `nn.Transformer*` expanded into attention, residuals and FFN | — | — |
| fallback when tracing fails | linear | linear | — |
| weights left untouched (recurrent, bare parameters) | listed in the report | listed in the report | listed in the report |
| parameters | in place | in place | functional, returns a new tree |

## Same weights everywhere

Weights are drawn in NumPy from the seed, and each backend only translates the layout. The
same architecture and seed give **bit-identical weights in all three frameworks**. See the
[tutorial](../tutorials/multiple-frameworks.md).
