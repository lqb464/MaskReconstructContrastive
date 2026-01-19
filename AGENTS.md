# AGENTS.md
# Codex Working Agreement

This repository is large and contains multiple independent pipelines.
Codex must strictly follow the edit scope and task rules below.

Current focus:
- Alzheimer classifier training using Swin-UNet encoder features
- Encoder-only, supervised setting
- Logic is finalized in chat first, then implemented by Codex

---

## 1. Edit Scope (Hard Rule)

Codex is ONLY allowed to modify files in the following locations:

- swin_unet/src/ver2/alzheimer_classifier/**
- swin_unet/src/ver2/train_alzheimer_classifier.py  
  (wrapper only, no training logic)

Everything else is **read-only**, including:
- preprocessing/**
- swin_unet/scripts/**
- swin_unet/outputs/**
- unet/**
- all other files under swin_unet/src/**
- root files (README.md, TODO.md, LICENSE)

Rules:
- Do not edit read-only files.
- Do not suggest edits to read-only files.
- If a required change would touch a read-only file, STOP and explain why.

---

## 2. Repo Context (Relevant Part Only)

- swin_unet/src/ver2/
  - main.py: SSL training entry (out of scope)
  - trainer.py: SSL trainer with recon/contrastive logic (out of scope)
  - models/: SwinUNetDualViewSSL and encoder utilities
  - data/: SSL dataset logic (out of scope unless explicitly allowed)
  - train_alzheimer_classifier.py: entry point for classifier training (wrapper)
  - alzheimer_classifier/: **new dedicated classifier package**

The classifier pipeline is **separate** from SSL training.

---

## 3. Model and Training Constraints

### Encoder-Only Mode
- Use encoder features only.
- Do not enable reconstruction.
- Do not enable contrastive learning.
- Do not create or load projection heads.
- Do not compute contrastive losses.

### Input Contract (Must Match Original Behavior)
- Each sample produces **two views**:
  - view 1: original image
  - view 2: flipped image
- **No masking** is applied.
- Inputs must match what the original encoder expects
  (shape, dtype, normalization).

### Supervised Requirement
- Labels are mandatory.
- Training must fail fast if labels are missing.
- Classifier head output dimension must match number of classes.

---

## 4. K-Fold Training Requirements

- Training must support **K-fold cross-validation**.
- Splitting must be **multi-class safe**:
  - Use stratified splitting by label by default.
- For each fold:
  - train
  - validate
  - test
  - record metrics

Metrics per fold must include (at minimum):
- validation metrics
- test metrics

At the end:
- Aggregate metrics across folds (mean and std).
- Save per-fold and aggregate results to disk.

---

## 5. Code Organization Rules

- All classifier logic lives in:
  - swin_unet/src/ver2/alzheimer_classifier/
- `train_alzheimer_classifier.py` must be a **thin wrapper** only:
  - parse args
  - call alzheimer_classifier entry
  - no training logic

If the code grows large:
- Refactor using helper functions or modules **inside the folder only**.
- Do not create new top-level folders elsewhere.

---

## 6. Logging and Safety Checks

- Add explicit checks that fail fast when:
  - labels are missing
  - masking is enabled
  - contrastive or projection heads are enabled
  - class count mismatches classifier output

- Logging should be deterministic and minimal.
- Do not add progress bars unless explicitly requested.

---

## 7. Workflow Rule (Very Important)

- We finalize **logic and behavior in chat first**.
- Codex implements only what is already agreed.
- Codex must not redesign or reinterpret requirements.

If any instruction is unclear:
- STOP and ask for clarification.

---

## 8. Definition of Done

- Classifier training runs using encoder-only mode.
- Dual-view input (original + flipped) is enforced.
- No masking, no contrastive logic anywhere.
- K-fold training works for multi-class datasets.
- Per-fold and aggregate metrics are saved.
- No files outside the allowed scope are modified.
