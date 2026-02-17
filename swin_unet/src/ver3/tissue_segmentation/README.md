# Tissue Segmentation (ver3)

Multi-class tissue segmentation task on the ver3 Swin-UNet stack.
This task predicts categorical logits and trains with `CrossEntropyLoss`.

## How to run

### Quick start (minimum practical command)
```bash
python -m swin_unet.src.ver3.cli train-tissue \
  --train-root /path/to/images/train \
  --eval-root /path/to/images/eval \
  --train-label /path/to/labels/train \
  --eval-label /path/to/labels/eval \
  --train-list /path/to/scans_test.txt \
  --eval-list /path/to/scans_valid.txt \
  --seg-labels /path/to/seg_labels.txt \
  --label-mode 1 \
  --image-ext .png \
  --label-suffix _label.npz \
  --image-size 192 \
  --epochs 200 \
  --batch-size 64 \
  --lr 3e-4 \
  --num-workers 4 \
  --amp \
  --pin-memory \
  --out-dir runs_ssl_swinunet \
  --run-name tissue_mode1
```

Optional alias (supported by `ver3/cli.py`):
```bash
python -m swin_unet.src.ver3.cli tissue ...
```

## Dataset and mapping assumptions
- Train set is resolved only from `--train-list` (use `scans_test.txt`).
- Eval set is resolved only from `--eval-list` (use `scans_valid.txt`).
- Label filename pattern is `<stem>_label.npz` (configurable by `--label-suffix`).
- Label lookup order:
  1. `<train-label|eval-label> / relative_image_parent / (<stem> + label_suffix)`
  2. fallback deterministic basename-stem match under the active label root
- Samples without labels are filtered unless `--strict-pairs 1` is set.

## Label encoding modes

| Mode | Semantics |
|---|---|
| `1` | Merge `unknown` + `non brain` into class `0`; remap all other labels to contiguous `1..K`. |
| `2` | Merge `unknown` + `non brain` into class `0`; keep all other labels at original ids. |
| `3` | Keep `unknown` and `non brain` distinct; remap all labels to contiguous `0..K-1`. |
| `4` | Keep labels unchanged (no merge, no remap). |

`--num-classes` defaults to `0` (infer). For keep-id modes (`2` and `4`), set it only when you need an explicit larger output space.

## Metrics
- Per-class Dice is computed from `argmax(logits, dim=1)`.
- Class IDs considered for Dice are restricted to encoded IDs derived from `seg_labels.txt` mapping.
  - This avoids sparse keep-id gaps (for example, missing ids between `0..max_id`) affecting macro Dice.
- Macro Dice is computed from per-class Dice:
  - default excludes background class `0`
  - include background with `--dice-include-bg`
- Empty-class handling (epoch-level denominator = 0):
  - default excludes those classes from macro
  - use `--dice-empty-as-one` to treat them as Dice = `1.0`
- Current CLI has no `--dice-exclude-ids` flag.
- Best checkpoint criterion: maximize `eval_macro_dice`.

## CLI reference

### Data paths
| Flag | Type | Default | Description |
|---|---|---|---|
| `--train-root` | `str` | required | Root directory of train input images. |
| `--eval-root` | `str` | required | Root directory of eval input images. |
| `--train-label` | `str` | required | Root directory of train segmentation labels. |
| `--eval-label` | `str` | required | Root directory of eval segmentation labels. |
| `--train-list` | `str` | required | Train scan list path (`scans_test.txt`). |
| `--eval-list` | `str` | required | Eval scan list path (`scans_valid.txt`). |
| `--seg-labels` | `str` | `swin_unet/src/ver3/tissue_segmentation/txt/seg_labels.txt` | Path to `seg_labels.txt`. |
| `--image-ext` | `str` | `.png` | Image extension used for indexing. |
| `--label-suffix` | `str` | `_label.npz` | Label filename suffix. |
| `--label-key` | `str` | `""` | Optional NPZ key for label array. |
| `--data-root` | `str` | `""` | Inherited field; tissue main binds this from `--train-root` when empty. |
| `--image-size` | `int` | `192` | Inherited resize size. |
| `--target-size` | `int` | `0` | Task-specific square resize override (`0` uses `--image-size`). |
| `--resize-mode` | `str` | `letterbox` | Pair resize strategy: `letterbox` or `direct`. |
| `--plane` | `str` | `auto` | `axial`/`coronal` fixed plane, or `auto` to infer from file path/name tokens (matches `axial` / `coronal`). |
| `--strict-pairs` | `int` | `0` | `1`: error on missing label. `0`: filter unlabeled samples. |
| `--preprocessed-dir` | `str` | `""` | Inherited field (accepted). |
| `--skip-resize-in-loader` / `--no-skip-resize-in-loader` | bool toggle | off | Inherited loader resize toggle (accepted). |
| `--label-csv` | `str` | `""` | Inherited label CSV field (accepted). |
| `--label-path-col` | `str` | `image_path` | Inherited label CSV key (accepted). |
| `--label-col` | `str` | `label` | Inherited label CSV key (accepted). |

### Label encoding
| Flag | Type | Default | Description |
|---|---|---|---|
| `--label-mode` | `int` | `1` | Encoding mode (`1`, `2`, `3`, `4`). |
| `--num-classes` | `int` | `0` | Output classes override (`0` means infer from mapping). |
| `--strict-label-ids` / `--no-strict-label-ids` | bool toggle | strict on | Require all label ids in files to exist in `seg_labels`. |
| `--allow-unknown-label-ids` / `--no-allow-unknown-label-ids` | bool toggle | off | Only valid with non-strict ids; maps unknown ids to class `0`. |
| `--ignore-index` | `int` | `-100` | `CrossEntropyLoss(ignore_index=...)`. |

### Training
| Flag | Type | Default | Description |
|---|---|---|---|
| `--epochs` | `int` | `200` | Number of epochs. |
| `--batch-size` | `int` | `64` | Batch size. |
| `--lr` | `float` | `3e-4` | Learning rate. |
| `--weight-decay` | `float` | `1e-4` | AdamW weight decay. |
| `--ce-class-weights` | `str` | `""` | Optional comma-separated CE class weights (length must equal `num_classes`). |
| `--num-workers` | `int` | `4` | DataLoader workers. |
| `--pin-memory` / `--no-pin-memory` | bool toggle | on | DataLoader pin memory. |
| `--drop-last` / `--no-drop-last` | bool toggle | on | Drop last train batch. |
| `--amp` / `--no-amp` | bool toggle | on | Mixed precision toggle. |
| `--cpu` | flag | off | Force CPU device. |
| `--seed` | `int` | `42` | RNG seed. |
| `--grad-clip` | `float` | `1.0` | Gradient clipping max norm. |
| `--warmup-epochs` | `int` | `5` | Scheduler warmup epochs. |
| `--min-lr` | `float` | `1e-6` | Scheduler minimum LR. |
| `--val-every` | `int` | `1` | Run eval every N epochs. |
| `--resume-ckpt` | `str` | `""` | Inherited checkpoint load path (accepted). |
| `--ckpt-load-mode` | `str` | `none` | Inherited load mode: `none`, `full`, `encoder_only`. |
| `--freeze-encoder-epochs` | `int` | `0` | Inherited freeze setting (accepted). |
| `--reset-proj-head` / `--no-reset-proj-head` | bool toggle | on | Inherited projection-head reset toggle (accepted). |
| `--enable-reconstruct` / `--disable-reconstruct` | bool toggle | on | Accepted; tissue main enforces reconstruct path on. |
| `--enable-contrastive` / `--disable-contrastive` | bool toggle | off (tissue default) | Accepted; tissue run enforces contrastive off. |
| `--single-view` / `--dual-view` | bool toggle | base parser default is dual-view | Accepted; tissue run enforces single-view. |

### Logging/outputs
| Flag | Type | Default | Description |
|---|---|---|---|
| `--out-dir` | `str` | `runs_ssl_swinunet` | Root output directory. |
| `--run-name` | `str` | `""` | Optional run subdirectory. |
| `--ckpt-dir` | `str` | `""` | Inherited field (accepted). Tissue outputs checkpoints under `out_dir[/run_name]/checkpoints`. |
| `--save-latest-every` | `int` | `1` | Save `latest.pt` every N epochs. |
| `--save-best-after-epoch` | `int` | `0` | Earliest epoch eligible for best-checkpoint updates. |
| `--save-best-every` | `int` | `1` | Best-checkpoint comparison cadence. |

`epoch_log.csv` includes core train/eval Dice/loss columns plus context fields:
`train_num_excluded_classes`, `eval_num_excluded_classes`, `eval_ran`,
`best_updated`, `best_eval_macro_dice`, `train_time_s`, `eval_time_s`, `epoch_time_s`.

### Metrics
| Flag | Type | Default | Description |
|---|---|---|---|
| `--dice-include-bg` / `--no-dice-include-bg` | bool toggle | exclude bg | Include class `0` in macro Dice reduction. |
| `--dice-empty-as-one` / `--no-dice-empty-as-one` | bool toggle | exclude empty classes | Empty-class macro behavior. |

### Visualization
| Flag | Type | Default | Description |
|---|---|---|---|
| `--vis-every` | `int` | `20` | Save eval visualization every N epochs. |
| `--vis-num` | `int` | `4` | Number of eval samples rendered in tissue grid. |
| `--vis-threshold` | `float` | `0.5` | Reserved tissue visualization parameter. |
| `--vis-n-results` | `int` | `8` | Inherited field (accepted). |
| `--no-tqdm` | `int` | `0` | `1` disables tqdm progress bars. |
| `--debug-shapes` | `int` | `0` | `1` prints shape debug logs for early samples. |

### Inherited model and auxiliary flags (accepted by parser)
| Flag | Type | Default | Description |
|---|---|---|---|
| `--in-ch` | `int` | `1` | Input channels. |
| `--patch-size` | `int` | `16` | Swin patch size. |
| `--mask-ratio` | `float` | `0.35` | Inherited mask ratio field (accepted). |
| `--enable-masking` / `--disable-masking` | bool toggle | on in base parser | Accepted; tissue run sets masking off. |
| `--embed-dim` | `int` | `96` | Embedding dimension. |
| `--enc-depths` | `int x4` | `2 2 6 2` | Encoder depths. |
| `--dec-depths` | `int x3` | `6 2 2` | Decoder depths. |
| `--num-heads` | `int x4` | `3 6 12 24` | Attention heads. |
| `--window-size` | `int` | `7` | Swin window size. |
| `--proj-dim` | `int` | `128` | Inherited projection dim. |
| `--bottleneck-dim` | `int` | `256` | Inherited bottleneck dim. |
| `--split-to-stage` | `int` | `1` | Inherited dual-view split stage. |
| `--shared-from-stage` | `int` | `2` | Inherited dual-view shared stage. |
| `--plane-inject-method` | `str` | `film` | Plane conditioning method: `film` or `add`. |
| `--enable_saca` | flag | off | Enable SACA. |
| `--saca_position` | `str` | `after_stage1` | Single SACA placement. |
| `--saca_positions` | `str` | `""` | Comma-separated SACA placements. |
| `--saca_gate_init` | `float` | `0.0` | SACA gate initialization. |
| `--saca_warmup_epochs` | `int` | `5` | SACA warmup epochs. |
| `--ramp-contrastive` | `int` | `20` | Inherited schedule field (accepted). |
| `--enable-masked-loss` | flag | off | Inherited field (accepted). |
| `--recon-loss` | `str` | `weighted_bce_logits` | Inherited reconstruction loss field (accepted). |
| `--fg-eps` | `float` | `0.02` | Inherited field (accepted). |
| `--fg-weight` | `float` | `10.0` | Inherited field (accepted). |
| `--lambda-recon` | `float` | `0.0` | Inherited loss weight (accepted). |
| `--lambda-contrast` | `float` | `0.0` | Inherited loss weight (accepted). |
| `--temperature` | `float` | `0.2` | Inherited field (accepted). |
| `--dice-loss-weight` | `float` | `0.2` | Inherited field (accepted). |
| `--dice-mode` | `str` | `fg` | Inherited field (accepted). |
| `--dice-smooth` | `float` | `1e-6` | Inherited field (accepted). |
| `--aug-p-noise` | `float` | `0.7` | Inherited augmentation field (accepted). |
| `--aug-p-jitter` | `float` | `0.7` | Inherited augmentation field (accepted). |
| `--aug-p-blur` | `float` | `0.2` | Inherited augmentation field (accepted). |
| `--aug-noise-std` | `float` | `0.02` | Inherited augmentation field (accepted). |
| `--aug-jitter-strength` | `float` | `0.1` | Inherited augmentation field (accepted). |
| `--aug-blur-kernel` | `int` | `3` | Inherited augmentation field (accepted). |
| `--contrastive_loss_type` | `str` | `infonce` | Inherited field (accepted). |
| `--contrastive_position` | `str` | `bottleneck` | Inherited field (accepted). |
| `--vicreg_invariance_weight` | `float` | `25.0` | Inherited field (accepted). |
| `--vicreg_variance_weight` | `float` | `25.0` | Inherited field (accepted). |
| `--vicreg_covariance_weight` | `float` | `1.0` | Inherited field (accepted). |
| `--vicreg_variance_eps` | `float` | `1e-4` | Inherited field (accepted). |
| `--vicreg_target_std` | `float` | `1.0` | Inherited field (accepted). |
| `--enable-tsne` | flag | off | Inherited field (accepted). |
| `--tsne-only-if-labeled` / `--tsne-even-if-unlabeled` | bool toggle | only-if-labeled on | Inherited field (accepted). |
| `--tsne-every` | `int` | `20` | Inherited field (accepted). |
| `--tsne-max-items` | `int` | `1000` | Inherited field (accepted). |

## Example commands

### Mode 1
```bash
python -m swin_unet.src.ver3.cli train-tissue \
  --train-root /data/images/train \
  --eval-root /data/images/eval \
  --train-label /data/labels/train \
  --eval-label /data/labels/eval \
  --train-list /data/splits/scans_test.txt \
  --eval-list /data/splits/scans_valid.txt \
  --seg-labels /data/splits/seg_labels.txt \
  --label-mode 1 \
  --image-ext .png \
  --label-suffix _label.npz \
  --out-dir runs_ssl_swinunet \
  --run-name tissue_mode1
```

### Mode 2
```bash
python -m swin_unet.src.ver3.cli train-tissue \
  --train-root /data/images/train \
  --eval-root /data/images/eval \
  --train-label /data/labels/train \
  --eval-label /data/labels/eval \
  --train-list /data/splits/scans_test.txt \
  --eval-list /data/splits/scans_valid.txt \
  --seg-labels /data/splits/seg_labels.txt \
  --label-mode 2 \
  --num-classes 106 \
  --image-ext .png \
  --label-suffix _label.npz \
  --out-dir runs_ssl_swinunet \
  --run-name tissue_mode2
```

### Mode 3
```bash
python -m swin_unet.src.ver3.cli train-tissue \
  --train-root /data/images/train \
  --eval-root /data/images/eval \
  --train-label /data/labels/train \
  --eval-label /data/labels/eval \
  --train-list /data/splits/scans_test.txt \
  --eval-list /data/splits/scans_valid.txt \
  --seg-labels /data/splits/seg_labels.txt \
  --label-mode 3 \
  --image-ext .png \
  --label-suffix _label.npz \
  --out-dir runs_ssl_swinunet \
  --run-name tissue_mode3
```

### Mode 4
```bash
python -m swin_unet.src.ver3.cli train-tissue \
  --train-root /data/images/train \
  --eval-root /data/images/eval \
  --train-label /data/labels/train \
  --eval-label /data/labels/eval \
  --train-list /data/splits/scans_test.txt \
  --eval-list /data/splits/scans_valid.txt \
  --seg-labels /data/splits/seg_labels.txt \
  --label-mode 4 \
  --num-classes 106 \
  --image-ext .png \
  --label-suffix _label.npz \
  --out-dir runs_ssl_swinunet \
  --run-name tissue_mode4
```

### Include background in macro Dice
```bash
python -m swin_unet.src.ver3.cli train-tissue \
  --train-root /data/images/train \
  --eval-root /data/images/eval \
  --train-label /data/labels/train \
  --eval-label /data/labels/eval \
  --train-list /data/splits/scans_test.txt \
  --eval-list /data/splits/scans_valid.txt \
  --seg-labels /data/splits/seg_labels.txt \
  --label-mode 1 \
  --dice-include-bg \
  --out-dir runs_ssl_swinunet \
  --run-name tissue_with_bg_macro
```

### Disable strict label id checks
```bash
python -m swin_unet.src.ver3.cli train-tissue \
  --train-root /data/images/train \
  --eval-root /data/images/eval \
  --train-label /data/labels/train \
  --eval-label /data/labels/eval \
  --train-list /data/splits/scans_test.txt \
  --eval-list /data/splits/scans_valid.txt \
  --seg-labels /data/splits/seg_labels.txt \
  --label-mode 1 \
  --no-strict-label-ids \
  --allow-unknown-label-ids \
  --out-dir runs_ssl_swinunet \
  --run-name tissue_relaxed_ids
```
