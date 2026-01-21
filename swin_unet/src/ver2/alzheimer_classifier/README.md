# Alzheimer Classifier Package

<!--
Summary (short):
- Structure: __init__.py, cli.py, io.py, train.py (metrics computed in train.py)
- Run: python -m alzheimer_classifier.main [args]
- Inputs: two views per sample (original + flipped), labels required, no masking, encoder-only
- Outputs: best/latest checkpoints and metrics json under out_dir
- Loss: focal (default) or weighted cross entropy (wce)
-->

<!--
Details:
- Run: python -m alzheimer_classifier.main --out_dir runs/alzheimer_cls
- Inputs: PIL images -> grayscale -> resize -> tensor; returns (x1, x2, y)
- Loss:
  - Focal (default): --loss_type focal --focal_gamma 1.0 --focal_alpha list:w0,w1,w2,w3
    - focal_alpha formats: scalar:0.25 or list:w0,w1,w2,w3
  - Weighted CE: --loss_type wce --ce_class_weights list:w0,w1,w2,w3
    - ce_class_weights format: list:w0,w1,w2,w3 (required for wce)
- Outputs:
  - out_dir/checkpoints/best_cls.pt
  - out_dir/checkpoints/latest_cls.pt
  - out_dir/metrics/single_split_metrics.json
-->
---

Usage examples

Focal loss (recommended defaults; provide your own alpha list):

```
python -m alzheimer_classifier.main \
  --out_dir runs/alzheimer_cls \
  --loss_type focal \
  --focal_gamma 1.0 \
  --focal_alpha list:1,1,1,1
```

Weighted cross entropy (provide your own weights list):

```
python -m alzheimer_classifier.main \
  --out_dir runs/alzheimer_cls \
  --loss_type wce \
  --ce_class_weights list:1,1,1,1
```

Train

| Class              | Train samples |
| ------------------ | ------------- |
| Non-Demented       | 2,560         |
| Very Mild Demented | 1,792         |
| Mild Demented      | 717           |
| Moderate Demented  | 51            |
| **Total**           | **5,120**     |


---

Test

| Class              | Test samples |
| ------------------ | ------------ |
| Non-Demented       | 640          |
| Very Mild Demented | 448          |
| Mild Demented      | 179          |
| Moderate Demented  | 13           |
| **Total**           | **1,280**    |
