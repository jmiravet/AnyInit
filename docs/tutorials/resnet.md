# A ResNet on CIFAR-10

Trains a ResNet on CIFAR-10 with AnyInit and with the stock initialization. The data is
downloaded into `./data` on first run, and CUDA is used when available.

In a residual block the identity path already carries signal that no weight scales, so the
objective after the addition can be out of reach for the layers on the branch. AnyInit
drives those scales toward their floor and says so in the report.

```bash
python examples/resnet_cifar.py --epochs 3
```

```python
--8<-- "examples/resnet_cifar.py"
```
