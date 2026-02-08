# ver3 Run Guide

This `ver3` folder keeps legacy entrypoints and adds a unified CLI.

## Unified CLI

Primary entrypoint:

```bash
python -m swin_unet.src.ver3.cli --help
```

Subcommands:

- `train-ssl` (same as `swin_unet.src.ver3.main`)
- `eval-ssl` (same as `swin_unet.src.ver3.eval`)
- `train-cls` (same as `swin_unet.src.ver3.alzheimer_classifier.main`)
- `train-mask` (same as `swin_unet.src.ver3.mask_reconstruction.main`)

Aliases:

- `train -> train-ssl`
- `eval -> eval-ssl`
- `cls -> train-cls`
- `mask -> train-mask`

## Scripts To Run

Set your paths first:

```bash
export DATA_ROOT="/path/to/data_root"
export EVAL_ROOT="/path/to/eval_root"
export TRAIN_MASK_DIR="/path/to/mask_train"
export VAL_MASK_DIR="/path/to/mask_val"
```

SSL training (reconstruction-only baseline):

```bash
python -m swin_unet.src.ver3.cli train-ssl \
  --data-root "$DATA_ROOT" \
  --enable-reconstruct \
  --disable-contrastive \
  --disable-masking \
  --lambda-recon 1.0 \
  --epochs 20 \
  --batch-size 16 \
  --image-size 192 \
  --out-dir runs_ssl_swinunet \
  --run-name ssl_recon_baseline
```

SSL training (reconstruction + contrastive):

```bash
python -m swin_unet.src.ver3.cli train-ssl \
  --data-root "$DATA_ROOT" \
  --enable-reconstruct \
  --enable-contrastive \
  --lambda-recon 1.0 \
  --lambda-contrast 1.0 \
  --contrastive_loss_type infonce \
  --contrastive_position bottleneck \
  --epochs 20 \
  --batch-size 16 \
  --out-dir runs_ssl_swinunet \
  --run-name ssl_joint
```

SSL eval from a concrete checkpoint path:

```bash
python -m swin_unet.src.ver3.cli eval-ssl \
  --ckpt runs_ssl_swinunet/ssl_joint/checkpoints/best.pt \
  --data-root "$EVAL_ROOT" \
  --batch-size 16 \
  --save-vis
```

SSL eval using `best` with a run directory:

```bash
python -m swin_unet.src.ver3.cli eval-ssl \
  --ckpt best \
  --run-dir runs_ssl_swinunet/ssl_joint \
  --data-root "$EVAL_ROOT"
```

Alzheimer classifier training:

```bash
python -m swin_unet.src.ver3.cli train-cls \
  --out_dir runs/alzheimer_cls/exp1 \
  --epochs 20 \
  --batch_size 32 \
  --classification_mode classification_default \
  --feature_level bottleneck \
  --fusion avg \
  --loss_type focal
```

Alzheimer classifier training with pretrained SSL encoder:

```bash
python -m swin_unet.src.ver3.cli train-cls \
  --out_dir runs/alzheimer_cls/exp_pretrained \
  --resume-ckpt runs_ssl_swinunet/ssl_joint/checkpoints/best.pt \
  --ckpt-load-mode encoder_only \
  --epochs 20 \
  --batch_size 32 \
  --loss_type focal
```

Mask reconstruction training:

```bash
python -m swin_unet.src.ver3.cli train-mask \
  --train_dir "$TRAIN_MASK_DIR" \
  --val_dir "$VAL_MASK_DIR" \
  --enable-reconstruct \
  --disable-contrastive \
  --disable-masking \
  --lambda-recon 1.0 \
  --batch-size 16 \
  --epochs 20 \
  --out-dir runs_mask \
  --run-name mask_recon_exp
```

Mask reconstruction offline preprocessing + fast loader path:

```bash
python -m swin_unet.src.ver3.tools.preprocess_mask_dataset \
  --input-dir "$TRAIN_MASK_DIR" \
  --output-dir /data_preprocessed/mask/train \
  --image-size 256 \
  --ext .png \
  --output-mask-suffix _mask.npy \
  --resize-mode letterbox \
  --num-workers 8

python -m swin_unet.src.ver3.cli train-mask \
  --preprocessed_dir /data_preprocessed/mask/train \
  --val_dir /data_preprocessed/mask/val \
  --image-size 256 \
  --skip_resize_in_loader \
  --enable-reconstruct \
  --disable-contrastive \
  --disable-masking \
  --batch-size 16 \
  --epochs 20 \
  --out-dir runs_mask \
  --run-name mask_recon_preprocessed
```

Smoke checks:

```bash
python scripts/smoke_ver3_imports.py
bash scripts/smoke_ver3_cli_help.sh
```

## Important CLI

### General

- `train-ssl`, `eval-ssl`, and `train-mask` mostly use `kebab-case` flags like `--batch-size`.
- `train-cls` keeps legacy `snake_case` flags like `--batch_size` and `--out_dir`.
- Keep this difference in mind when switching commands.

### `train-ssl` (main training)

- Mode flags: `--enable-reconstruct/--disable-reconstruct`, `--enable-contrastive/--disable-contrastive`.
- Weight constraints: set `--lambda-recon > 0` when reconstruction is enabled, and `--lambda-contrast > 0` when contrastive is enabled.
- Single-view rule: `--single-view` requires reconstruction on, contrastive off, and SACA off.
- Checkpoint loading: `--resume-ckpt`, `--ckpt-load-mode {none,full,encoder_only}`, optional `--reset-proj-head`.
- Loss controls: `--recon-loss`, `--enable-masked-loss`, `--dice-loss-weight`, `--dice-mode`, `--dice-smooth`.

### `eval-ssl`

- Checkpoint selection: use `--ckpt PATH`, or use `--ckpt best|latest` together with `--run-dir`.
- Dataset root: `--data-root` points to the evaluation dataset folder.
- Output toggles: `--save-vis` for reconstruction grids, `--enable-tsne` for t-SNE export.

### `train-cls`

- Architecture controls: `--classification_mode`, `--feature_level`, `--fusion`.
- View/SACA compatibility: use `--view_mode two|one_v1|one_v2`; if `--enable_saca`, keep `--view_mode two`.
- Loss setup: `--loss_type focal|wce|ce`; for `wce`, provide `--ce_class_weights list:w0,w1,...`; for focal, tune `--focal_alpha` and `--focal_gamma`.
- Pretrained loading: `--resume-ckpt`, `--ckpt-load-mode {none,full,encoder_only}`, optional `--freeze-recon` and `--freeze-decoder-recon`.

### `train-mask`

- Required dataset arg: `--train_dir`.
- Validation mode: `--val_dir` for explicit validation folder, otherwise internal split via `--val-ratio`.
- Pair I/O flags: `--image_ext`, `--mask_suffix`, `--mask_key`, `--strict_pairs`.
- Resize/debug flags: `--target-size`, `--resize-mode`, `--debug-shapes`.
- Preprocessed fast-path flags: `--preprocessed_dir`, `--skip_resize_in_loader`.
- Metadata checks: if `preprocess_meta.json` exists, loader validates preprocessed image size against runtime `--image-size`.
- Behavior guardrail: this entrypoint enforces reconstruction-only (no contrastive, no masking).

Mask reconstruction sanity checklist:
- Pair folder structure is consistent (image + mask suffix by stem).
- Preprocessed datasets include `preprocess_meta.json`.
- Preprocessed `H,W` matches runtime `--image-size` (current mask model expects square).
- Expected loader normalization remains `input/255.0` and `target/255.0` unless `--binarize-target` is used.

## Legacy Entrypoints

These still work:

```bash
python -m swin_unet.src.ver3.main --help
python -m swin_unet.src.ver3.eval --help
python -m swin_unet.src.ver3.alzheimer_classifier.main --help
python -m swin_unet.src.ver3.mask_reconstruction.main --help
```
