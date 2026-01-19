# Alzheimer Classifier Package

<!--
Summary (short):
- Structure: __init__.py, cli.py, io.py, kfold.py, train.py (metrics computed in train.py)
- Run: python train_alzheimer_classifier.py [args]
- K-fold: set --k_folds <N> --val_ratio <r> --seed <seed>
- Inputs: two views per sample (original + flipped), labels required, no masking, encoder-only
- Outputs: per-fold checkpoints/plots and metrics json under out_dir
-->

<!--
Details:
- Run: python train_alzheimer_classifier.py --out_dir runs/alzheimer_cls
- Enable K-fold: add --k_folds 5 --val_ratio 0.1 --seed 42
- Inputs: PIL images -> grayscale -> resize -> tensor; returns (x1, x2, y)
- Outputs:
  - out_dir/fold_XX/checkpoints/*
  - out_dir/fold_XX/plots/*
  - out_dir/metrics/fold_metrics.json
  - out_dir/metrics/metrics_summary.json (single-split uses single_split_metrics.json)
-->
