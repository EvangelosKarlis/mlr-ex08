| variant | change vs the baseline | parameters | train acc | train loss | validation | camera-like | last year (OOD) | best epoch, val |
|---|---|---|---|---|---|---|---|---|
| `baseline` | Task 3d network: the fixed baseline | 0.21 M | 0.719 | 0.746 | 0.595 ±0.054 | 0.393 ±0.018 | 0.467 ±0.067 | 0.631 |
| `ref-plain` | baseline WITHOUT augmentation (context for the baseline row) | 0.21 M | 0.970 | 0.162 | 0.500 ±0.054 | 0.391 ±0.033 | 0.497 ±0.012 | 0.607 |
| `v1-color` | ONE change: ColorJitter added to the augmentation | 0.21 M | 0.591 | 1.008 | 0.536 ±0.054 | 0.485 ±0.033 | 0.411 ±0.067 | 0.583 |
| `v2a-frozen` | ONE change: MobileNet with only the new last layer trained | 1.52 M | 0.964 | 0.157 | 0.702 ±0.036 | 0.670 ±0.029 | 0.606 ±0.033 | 0.762 |
| `v2b-finetune` | ONE change: MobileNet, whole network fine-tuned | 1.52 M | 0.944 | 0.180 | 0.702 ±0.125 | 0.603 ±0.056 | 0.594 ±0.050 | 0.726 |
| `v2b-noaug` | v2b without augmentation = the recipe of leaf_model.onnx, the model now on the page | 1.52 M | 0.974 | 0.095 | 0.571 ±0.071 | 0.591 ±0.036 | 0.492 ±0.046 | 0.690 |
| `v3-schedule` | v2b + colour augmentation + CosineAnnealingLR + 40 epochs | 1.52 M | 0.967 | 0.126 | 0.690 ±0.107 | 0.705 ±0.027 | 0.625 ±0.046 | 0.774 |
| `v4a-dropout` | ONE change: Dropout(0.5) before the last layer | 0.21 M | 0.541 | 1.065 | 0.429 ±0.107 | 0.387 ±0.103 | 0.356 ±0.075 | 0.571 |
| `v4b-gap` | ONE change: global average pooling instead of flatten | 65 k | 0.680 | 0.853 | 0.512 ±0.054 | 0.403 ±0.056 | 0.439 ±0.075 | 0.583 |
| `v4c-adamw` | ONE change: AdamW with weight decay 1e-4 | 0.21 M | 0.733 | 0.737 | 0.607 ±0.071 | 0.388 ±0.016 | 0.453 ±0.050 | 0.643 |
| `v4d-smooth` | ONE change: label smoothing 0.1 | 0.21 M | 0.749 | 0.900 | 0.548 ±0.054 | 0.400 ±0.013 | 0.447 ±0.025 | 0.595 |
| `v4e-combined` | v4a+b+c+d together, and the weights of the best epoch are the ones kept | 65 k | 0.634 | 1.061 | 0.500 ±0.036 | 0.379 ±0.060 | 0.453 ±0.013 | 0.548 |
| `v5-history` | ONE change: baseline trained on our 101 + 700 historical images | 0.21 M | 0.782 | 0.539 | 0.643 ±0.036 | 0.490 ±0.027 | 0.792 ±0.088 | 0.702 |
