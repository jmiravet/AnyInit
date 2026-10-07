# Reading the report

`initialize` returns a report. Print it to see what was done:

```
AnyInit — backend=pytorch, mode=analytic, distribution=normal, center=False, graph=graph
  solve: 6 pass(es), largest objective gap 4.36e-03

  layer                  kind         activation                   fan          scale    pred    meas
  ─────────────────────────────────────────────────────────────────────────────────────────────────────
  conv1                  Conv2d       none                         147       0.082479  1.0000  0.9846
  bn1                    BatchNorm2d  relu→MaxPool2d                 -  gain=0.624688  1.0000  0.9915
  layer1.0.bn2           BatchNorm2d  add→relu                       -  gain=0.058547  1.0032  0.9943
  layer2.0.bn2           BatchNorm2d  add→relu                       -  gain=1.000000  1.0000  1.0001
  layer2.0.downsample.1  BatchNorm2d  add→relu                       -  gain=1.000000  1.0000  1.0001
  layer4.1.bn2           BatchNorm2d  add→relu→AdaptiveAvgPool2d     -  gain=1.413617  1.0000  0.9425
  fc                     Linear       none                         512       0.044194  1.0000  0.9873

Validation
  largest unexplained prediction/measurement gap: 0.00% at bn1  [ok]  (raw gap 0.85%, the rest is sampling noise)
```

## The layer table

- **activation** is the path from the layer to where its target is enforced, not just a
  name. `add→relu` on two rows means both normalizations feed the same post-addition ReLU
  and were scaled together.
- `relu→MaxPool2d` means the target is enforced after the pooling, where the next layer
  reads. A 3×3 max pool changes the second moment a lot, so aiming at the ReLU would leave
  the next block mis-scaled.
- **scale** is the weight's standard deviation, or `gain=` for a normalization layer. A
  small gain, such as `layer1.0.bn2`'s `0.059`, marks a residual block whose identity path
  already carries the target on its own.
- **pred** and **meas** are the predicted and measured second moments; the target is 1.

## Validation

The validation line compares the prediction with one measured forward pass and reports the
gap beyond sampling noise. It is a sanity check, never used to pick a scale. It runs when
you pass an `input_spec`.

## Warnings

Anything AnyInit could not do correctly is listed: weights it left untouched (recurrent,
bare parameters), activations that are [unstable at this depth](stability.md), and
assumptions the [analytic mode](modes.md) could not hold.

`report.assert_healthy()` raises on any of these, which is handy in CI.
