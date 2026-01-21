# AGENTS.md — Codex Implementation Guide (ver2 / Alzheimer Classifier)

## 0) Repository Context

This repository contains a SwinUNetDualViewSSL-based pipeline with multiple components:
- `models/` encoder implementation (SwinUNetDualViewSSL)
- `training/` shared checkpoint utilities
- `alzheimer_classifier/` a standalone training package for classification on HF dataset `Falah/Alzheimer_MRI`

Key folder for this task:
- `alzheimer_classifier/` (contains `cli.py`, `train.py`, `kfold.py`, `io.py`, `README.md`, `main.py`)

Project tree reference (partial):
- `alzheimer_classifier/*`
- `models/*`
- `training/*`

## 1) High-Level Goal

Refactor the Alzheimer classifier training package to match the benchmark rule:

- Dataset uses ONLY two splits: `train` and `test`
- Train on `train`
- Evaluate on `test` every epoch
- Select the **best epoch by maximum test macro-F1**
- Save best checkpoint and metrics

Additionally:
- Determine the **exact label-index ↔ class-name mapping** from the dataset metadata
- Use that mapping to apply **class weights** correctly
- Add **Weighted Cross Entropy** as an alternative loss
- Keep Focal Loss as an option and expose hyperparameters to tune imbalance handling

## 2) Critical Constraints (MUST FOLLOW)

### 2.1 Do NOT change
- Model architecture and core encoder code under `models/`
- Data semantics: dataset is still `Falah/Alzheimer_MRI` loaded via HuggingFace `datasets.load_dataset`
- Input format: still returns `(x1, x2, y)` where `x2` is a flipped view
- Evaluation metric definition: macro-F1 is computed by sklearn with `average="macro"`
- Overall training loop behavior: per-epoch train then evaluate

### 2.2 MUST change
- Remove k-fold logic because the benchmark requires train/test only
- Remove useless or dead code paths related to k-fold splitting and fold aggregation
- Training entry should be a single "train/test" process; no fold directories, no concatenation of train+test

### 2.3 Benchmark selection rule
- Best model is selected ONLY by **test macro-F1** (not val, not loss, not accuracy)

### 2.4 Safety / correctness
- NEVER guess label mapping. Always read from dataset metadata:
  - `ds["train"].features["label"].names` if available
- Always print a mapping table at runtime:
  - `index -> class_name`
- When applying weights/alpha, ensure it matches the label index order.

## 3) Implementation Phases

### Phase 1 — Remove k-fold and dead code
Files likely affected:
- `alzheimer_classifier/cli.py`
- `alzheimer_classifier/train.py`
- `alzheimer_classifier/kfold.py` (should be deleted or no longer imported)
- `alzheimer_classifier/README.md` (remove k-fold usage docs)

Required actions:
- Delete CLI arguments that only exist for k-fold:
  - `--k_folds`, `--val_ratio`, `--group_field`
- Remove functions and imports related to:
  - `make_kfold_splits`, `FoldSplit`
  - `datasets.concatenate_datasets`
  - `run_kfold` path and fold metrics aggregation
- Update `run(args)` to always run a single train/test pipeline.

Acceptance criteria:
- Running the entrypoint does not import or reference `kfold.py`
- Code path is single-split only.

### Phase 2 — Standard train/test training process
In `alzheimer_classifier/train.py`:
- Keep per-epoch loop:
  - train on train loader
  - eval on test loader
- Select best epoch by test macro-F1
- Save:
  - `checkpoints/best_cls.pt`
  - `checkpoints/latest_cls.pt`
- Save metrics json under:
  - `out_dir/metrics/single_split_metrics.json`

Important:
- Do NOT create fold subdirectories.

Acceptance criteria:
- Best checkpoint corresponds to max test macro-F1 observed.
- Metrics JSON includes at minimum: best_epoch, best_score, test macro-F1.

### Phase 3 — Label mapping + Weighted CE + Loss selection
Add a configurable loss type via CLI:
- `--loss_type` with choices: `focal`, `wce`
- Weighted Cross Entropy should accept class weights from CLI:
  - `--ce_class_weights` format: `list:w0,w1,w2,w3` (size must match num_classes)
- Focal loss should keep:
  - `--focal_gamma`
  - `--focal_alpha` already supported

Runtime requirements:
- Print label mapping at startup:
  - `0: Non-Demented`
  - `1: Very Mild Demented`
  - `2: Mild Demented`
  - `3: Moderate Demented`
  (exact names depend on dataset metadata)
- Validate weight vector lengths match num_classes
- Ensure weights are on the correct device and correct dtype

Recommended default hyperparameters for imbalance (do not hardcode; expose via args):
- Focal: `gamma=1.0`, `alpha` list set by user (must match mapping)
- WCE: class weights set by user (must match mapping)

Acceptance criteria:
- Loss can be switched without code edits.
- Mapping and weight order is explicit and verified.

### Phase 4 — Clean docs + sanity logs
- Update `alzheimer_classifier/README.md`:
  - Remove k-fold instructions
  - Document new args: loss_type, ce_class_weights, focal params
- Add minimal, consistent logs:
  - per-epoch: train loss/acc, test loss/acc, test macro-F1
  - confusion matrix (optional but allowed)

Acceptance criteria:
- README accurately reflects the final pipeline.

## 4) Files and Code Touch Policy

Primary files to edit:
- `alzheimer_classifier/cli.py`
- `alzheimer_classifier/train.py`
- `alzheimer_classifier/README.md`
- Optionally delete: `alzheimer_classifier/kfold.py` (or leave but unused and not imported)

Do NOT edit unless required:
- `models/*`
- `training/*`
- shared SSL or recon code in `common/*`, `trainer.py`, etc.

## 5) Output Contract

After running:
- `out_dir/checkpoints/best_cls.pt`
- `out_dir/checkpoints/latest_cls.pt`
- `out_dir/metrics/single_split_metrics.json`

No fold-based outputs.

## 6) Common Failure Modes to Avoid

- Applying class weights with wrong index order
- Still selecting best epoch by val (must be test macro-F1)
- Leaving unused k-fold imports that break runtime
- Hardcoding label names instead of reading from dataset metadata
- Silent mismatch between num_classes and weight vector length

## 7) Quick Local Run Example

Example (Focal):
```bash
python -m alzheimer_classifier.main \
  --out_dir runs/alzheimer_cls \
  --loss_type focal \
  --focal_gamma 1.0 \
  --focal_alpha list:... \
  --batch_size 16 \
  --epochs 15
```

Example (Weighted CE):
```bash
python -m alzheimer_classifier.main \
  --out_dir runs/alzheimer_cls \
  --loss_type wce \
  --ce_class_weights list:w0,w1,w2,w3 \
  --batch_size 16 \
  --epochs 15
```