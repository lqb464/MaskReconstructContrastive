# Alzheimer Classifier (autoencoder ver3)

Classification on MAE/VAE bottleneck features (dual-view optional).

## Run

```bash
python -m autoencoder.src.cli train-cls \
  --mae \
  --image_size 256 \
  --resume-ckpt /path/to/ssl/checkpoints/best.pt \
  --ckpt-load-mode encoder_only \
  --classification_mode classification_bottleneck_concat \
  --loss_type focal \
  --epochs 20 \
  --batch_size 32 \
  --out_dir runs/alzheimer_cls_mae
```

## Supported modes

- `classification_default` — bottleneck GAP + fusion (`avg`/`concat`/`max`)
- `classification_bottleneck_concat` — concat view1/view2 bottleneck vectors

Stage1/stage2/multiscale modes from Swin-UNet are **not** supported.

## Outputs

- `out_dir/checkpoints/best_cls.pt`
- `out_dir/checkpoints/latest_cls.pt`
- `out_dir/epoch_log.csv`
- `out_dir/metrics/single_split_metrics.json`
