**Tissue Segmentation (ver3)**

Train Swin-UNet for multi-class tissue segmentation with fixed train/eval scan lists.

**Sample Contract**
- `input`: float32 `[1,H,W]` in `[0,1]`
- `target`: int64 `[H,W]` class ids after selected label mode
- `plane_one_hot`: float32 `[2]`
- `path`: original image path string

**Dataset Rules**
- Images are loaded from `--image-root`.
- Labels are loaded from a separate `--label-root` using `<stem>_label.npz` (default).
- Samples without labels are filtered out (or fail with `--strict-pairs 1`).
- No random split is used.
  - Train uses `--train-list` (fixed list, typically `scans_test.txt`)
  - Eval uses `--eval-list` (fixed list, typically `scans_valid.txt`)
  - No `val_ratio`, no random subset/split.

**Label Modes**
- `1`: unknown + non-brain -> `0`, others remapped to contiguous `1..K`
- `2`: unknown + non-brain -> `0`, others keep original ids
- `3`: no merge, all labels remapped to contiguous `0..K-1` (unknown/non-brain remain distinct)
- `4`: labels unchanged (identity)

**Metrics**
- Per-class Dice and Macro Dice are computed.
- Dice uses `argmax(logits)` predictions and class-wise formula `2|P∩T| / (|P|+|T|)`.
- Macro Dice excludes class `0` by default; set `--dice-include-bg` to include it.
- Classes with zero denominator over an epoch are excluded by default (`--no-dice-empty-as-one`).
  - Optional: `--dice-empty-as-one` treats those classes as Dice=1.0.

**Artifacts**
- `checkpoints/latest.pt`
- `checkpoints/best_eval_macro_dice.pt`
- `epoch_log.csv`
- Optional eval visualization under `vis/`
- Best checkpoint criterion: maximize `eval_macro_dice`.
