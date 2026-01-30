# AGENTS.md

## Objective

Optimize training speed for **reconstruction-only** mode without changing model semantics or training outcomes.

Scope is **performance refactor only**. No architecture change, no loss change, no numerical behavior change beyond floating point nondeterminism.

## Non-goals

* Do NOT enable single_view
* Do NOT remove dual-view path
* Do NOT change training logic, loss formulation, or model outputs
* Do NOT touch encoder freezing logic

---

## Confirmed Context

* Training mode: reconstruction loss only (contrastive disabled)
* Model: SwinUNetDualViewSSL v2
* Input image size is fixed per run
* Mask sampling happens **every iteration** via `sample_masks_anti_mirror`
* Attention masks in Swin blocks are recomputed every forward

---

## Phase 1: High-impact speed optimizations (MUST DO)

### 1. Cache Swin attention masks

**Files**:

* `models/model_utils.py`
* `models/swin_unet_dualview_ssl.py`

**Problem**:
`compute_attn_mask()` is recomputed every forward for identical `(H, W, window_size, shift_size, device)`.

**Action**:

* Introduce a global or static cache for attention masks
* Key by `(H, W, window_size, shift_size, device)`
* Reuse cached tensor if exists
* Preserve dtype and device

**Rules**:

* No logic change
* Cache must be safe for multi-epoch training

---

### 2. Optimize DataLoader throughput

**File**:

* `data/dataset.py`

**Action**:
When `num_workers > 0`, enable:

* `persistent_workers=True`
* `prefetch_factor=2`

Keep:

* existing shuffle logic
* existing pin_memory flag

---

### 3. Reduce Python overhead in training loop

**File**:

* `trainer.py`

**Action**:

* Replace per-iteration list appends for losses with running sum + counter
* Keep exact same logged mean values
* Disable per-iteration tqdm postfix unless explicitly enabled

---

### 4. Mask sampling optimization

**Files**:

* `training/batch_ops.py`
* `data/augmentation.py`

**Problem**:
`sample_masks_anti_mirror()` uses Python loops and `random.sample` per batch.

**Action**:

* Rewrite mask sampling using vectorized torch operations
* Eliminate Python loops over batch and patches
* Output tensor must be bitwise-compatible in shape and semantics

---

## Phase 2: Runtime-level optimizations (SAFE)

### 5. Enable cudnn benchmark

**File**:

* training entrypoint (`main.py` or `trainer.py`)

**Action**:
Set:

```python
torch.backends.cudnn.benchmark = True
```

Only when input size is fixed.

---

### 6. Optional torch.compile

**Action**:

* Wrap model with `torch.compile(model)` behind a config flag
* Default OFF

---

## Validation Checklist

Codex MUST ensure:

* Training loss curves remain numerically consistent
* No change in output tensor shapes
* No change in forward paths
* No removal of dual-view reconstruction
* No change to checkpoint format

---

## Deliverables

* Refactored code implementing Phase 1 and Phase 2
* No unrelated formatting or refactors
* Minimal diffs, clearly scoped

End of AGENTS.md
