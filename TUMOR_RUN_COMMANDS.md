# BraTS 2021 Tumor Segmentation Run Commands

This file contains the complete commands for running **10 different configurations** of the tumor segmentation task (`train-tumor`) on Kaggle.

---

## 📌 Shared Environment Paths & Parameters

Define these variables in your shell or notebook before running:

```bash
IMAGES=/kaggle/input/datasets/catcdunil/brats2021-extracted/images
LABELS=/kaggle/input/datasets/catcdunil/brats2021-extracted/labels
TRAIN_LIST=/kaggle/input/datasets/catcdunil/brats2021-extracted/train_list.txt
EVAL_LIST=/kaggle/input/datasets/catcdunil/brats2021-extracted/eval_list.txt
SEG_LABELS=/kaggle/working/MaskReconstructContrastive/swin_unet/src/tumor_segmentation/txt/seg_labels.txt
```

### Shared Configuration Arguments
- `--image-ext .png`
- `--label-suffix _label.npz`
- `--label-mode 3`
- `--ce-class-weights "1,5,3,8"`
- `--image-size 256`
- `--batch-size 32`
- `--epochs 60`
- `--plane axial`
- `--vis-every 1`
- `--out-dir runs`

---

## 🛠️ 10 Run Commands Configurations

### A. UNet Configurations

#### 1. UNet: Single View
```bash
!python -m swin_unet.src.cli train-tumor \
  --unet \
  --single-view \
  --train-root $IMAGES \
  --eval-root $IMAGES \
  --train-label $LABELS \
  --eval-label $LABELS \
  --train-list $TRAIN_LIST \
  --eval-list $EVAL_LIST \
  --seg-labels $SEG_LABELS \
  --image-ext .png \
  --label-suffix _label.npz \
  --label-mode 3 \
  --ce-class-weights "1,5,3,8" \
  --image-size 256 \
  --batch-size 32 \
  --epochs 60 \
  --plane axial \
  --vis-every 1 \
  --out-dir /kaggle/working/outputs \
  --run-name tumor_unet_single_view
```

#### 2. UNet: Dual View
```bash
!python -m swin_unet.src.cli train-tumor \
  --unet \
  --dual-view \
  --train-root $IMAGES \
  --eval-root $IMAGES \
  --train-label $LABELS \
  --eval-label $LABELS \
  --train-list $TRAIN_LIST \
  --eval-list $EVAL_LIST \
  --seg-labels $SEG_LABELS \
  --image-ext .png \
  --label-suffix _label.npz \
  --label-mode 3 \
  --ce-class-weights "1,5,3,8" \
  --image-size 256 \
  --batch-size 32 \
  --epochs 60 \
  --plane axial \
  --vis-every 1 \
  --out-dir /kaggle/working/outputs \
  --run-name tumor_unet_dual_view
```

---

### B. Swin-UNet Configurations

#### 3. Swin-UNet: Single View
```bash
!python -m swin_unet.src.cli train-tumor \
  --single-view \
  --train-root $IMAGES \
  --eval-root $IMAGES \
  --train-label $LABELS \
  --eval-label $LABELS \
  --train-list $TRAIN_LIST \
  --eval-list $EVAL_LIST \
  --seg-labels $SEG_LABELS \
  --image-ext .png \
  --label-suffix _label.npz \
  --label-mode 3 \
  --ce-class-weights "1,5,3,8" \
  --image-size 256 \
  --batch-size 32 \
  --epochs 60 \
  --plane axial \
  --vis-every 1 \
  --out-dir /kaggle/working/outputs \
  --run-name tumor_swin_single_view
```

#### 4. Swin-UNet: Dual View (No SACA)
```bash
!python -m swin_unet.src.cli train-tumor \
  --dual-view \
  --train-root $IMAGES \
  --eval-root $IMAGES \
  --train-label $LABELS \
  --eval-label $LABELS \
  --train-list $TRAIN_LIST \
  --eval-list $EVAL_LIST \
  --seg-labels $SEG_LABELS \
  --image-ext .png \
  --label-suffix _label.npz \
  --label-mode 3 \
  --ce-class-weights "1,5,3,8" \
  --image-size 256 \
  --batch-size 32 \
  --epochs 60 \
  --plane axial \
  --vis-every 1 \
  --out-dir /kaggle/working/outputs \
  --run-name tumor_swin_dual_view_no_saca
```

---

### C. Swin-UNet with Single SACA Configurations

#### 5. Swin-UNet: Single SACA (after_patch_embed)
```bash
!python -m swin_unet.src.cli train-tumor \
  --enable_saca \
  --saca_position after_patch_embed \
  --train-root $IMAGES \
  --eval-root $IMAGES \
  --train-label $LABELS \
  --eval-label $LABELS \
  --train-list $TRAIN_LIST \
  --eval-list $EVAL_LIST \
  --seg-labels $SEG_LABELS \
  --image-ext .png \
  --label-suffix _label.npz \
  --label-mode 3 \
  --ce-class-weights "1,5,3,8" \
  --image-size 256 \
  --batch-size 32 \
  --epochs 60 \
  --plane axial \
  --vis-every 1 \
  --out-dir /kaggle/working/outputs \
  --run-name tumor_swin_saca_after_patch_embed
```

#### 6. Swin-UNet: Single SACA (after_stage0)
```bash
!python -m swin_unet.src.cli train-tumor \
  --enable_saca \
  --saca_position after_stage0 \
  --train-root $IMAGES \
  --eval-root $IMAGES \
  --train-label $LABELS \
  --eval-label $LABELS \
  --train-list $TRAIN_LIST \
  --eval-list $EVAL_LIST \
  --seg-labels $SEG_LABELS \
  --image-ext .png \
  --label-suffix _label.npz \
  --label-mode 3 \
  --ce-class-weights "1,5,3,8" \
  --image-size 256 \
  --batch-size 32 \
  --epochs 60 \
  --plane axial \
  --vis-every 1 \
  --out-dir /kaggle/working/outputs \
  --run-name tumor_swin_saca_after_stage0
```

#### 7. Swin-UNet: Single SACA (after_merge0)
```bash
!python -m swin_unet.src.cli train-tumor \
  --enable_saca \
  --saca_position after_merge0 \
  --train-root $IMAGES \
  --eval-root $IMAGES \
  --train-label $LABELS \
  --eval-label $LABELS \
  --train-list $TRAIN_LIST \
  --eval-list $EVAL_LIST \
  --seg-labels $SEG_LABELS \
  --image-ext .png \
  --label-suffix _label.npz \
  --label-mode 3 \
  --ce-class-weights "1,5,3,8" \
  --image-size 256 \
  --batch-size 32 \
  --epochs 60 \
  --plane axial \
  --vis-every 1 \
  --out-dir /kaggle/working/outputs \
  --run-name tumor_swin_saca_after_merge0
```

#### 8. Swin-UNet: Single SACA (after_stage1)
```bash
!python -m swin_unet.src.cli train-tumor \
  --enable_saca \
  --saca_position after_stage1 \
  --train-root $IMAGES \
  --eval-root $IMAGES \
  --train-label $LABELS \
  --eval-label $LABELS \
  --train-list $TRAIN_LIST \
  --eval-list $EVAL_LIST \
  --seg-labels $SEG_LABELS \
  --image-ext .png \
  --label-suffix _label.npz \
  --label-mode 3 \
  --ce-class-weights "1,5,3,8" \
  --image-size 256 \
  --batch-size 32 \
  --epochs 60 \
  --plane axial \
  --vis-every 1 \
  --out-dir /kaggle/working/outputs \
  --run-name tumor_swin_saca_after_stage1
```

---

### D. Swin-UNet with Multi SACA Configurations

#### 9. Swin-UNet: Multi SACA (Đặt ở 2 lớp đầu)
Uses `after_patch_embed` and `after_stage0`.

```bash
!python -m swin_unet.src.cli train-tumor \
  --enable_saca \
  --saca_positions after_patch_embed,after_stage0 \
  --train-root $IMAGES \
  --eval-root $IMAGES \
  --train-label $LABELS \
  --eval-label $LABELS \
  --train-list $TRAIN_LIST \
  --eval-list $EVAL_LIST \
  --seg-labels $SEG_LABELS \
  --image-ext .png \
  --label-suffix _label.npz \
  --label-mode 3 \
  --ce-class-weights "1,5,3,8" \
  --image-size 256 \
  --batch-size 32 \
  --epochs 60 \
  --plane axial \
  --vis-every 1 \
  --out-dir /kaggle/working/outputs \
  --run-name tumor_swin_multi_saca_first_two
```

#### 10. Swin-UNet: Multi SACA (Đặt cả 4 lớp)
Uses `after_patch_embed`, `after_stage0`, `after_merge0`, and `after_stage1`.

```bash
!python -m swin_unet.src.cli train-tumor \
  --enable_saca \
  --saca_positions after_patch_embed,after_stage0,after_merge0,after_stage1 \
  --train-root $IMAGES \
  --eval-root $IMAGES \
  --train-label $LABELS \
  --eval-label $LABELS \
  --train-list $TRAIN_LIST \
  --eval-list $EVAL_LIST \
  --seg-labels $SEG_LABELS \
  --image-ext .png \
  --label-suffix _label.npz \
  --label-mode 3 \
  --ce-class-weights "1,5,3,8" \
  --image-size 256 \
  --batch-size 32 \
  --epochs 60 \
  --plane axial \
  --vis-every 1 \
  --out-dir /kaggle/working/outputs \
  --run-name tumor_swin_multi_saca_all_four
```

---

## 🚀 C. UNet with SACA Configurations (New)

Since UNet has shown superior performance on 2D slice segmentation, we can now apply Symmetric Cross-Attention (SACA) to UNet configurations. SACA maps to UNet blocks as follows:
- `after_patch_embed` -> after `enc1` (16 channels)
- `after_stage0` -> after `enc2` (32 channels)
- `after_merge0` -> after `enc3` (64 channels)
- `after_stage1` -> after `enc4` (128 channels)

#### 11. UNet: Single SACA (Đặt ở lớp after_patch_embed)
```bash
!python -m swin_unet.src.cli train-tumor \
  --unet \
  --enable_saca \
  --saca_position after_patch_embed \
  --train-root $IMAGES \
  --eval-root $IMAGES \
  --train-label $LABELS \
  --eval-label $LABELS \
  --train-list $TRAIN_LIST \
  --eval-list $EVAL_LIST \
  --seg-labels $SEG_LABELS \
  --image-ext .png \
  --label-suffix _label.npz \
  --label-mode 3 \
  --ce-class-weights "1,5,3,8" \
  --image-size 256 \
  --batch-size 32 \
  --epochs 60 \
  --plane axial \
  --vis-every 1 \
  --out-dir /kaggle/working/outputs \
  --run-name tumor_unet_saca_after_patch_embed
```

#### 12. UNet: Single SACA (Đặt ở lớp after_stage0)
```bash
!python -m swin_unet.src.cli train-tumor \
  --unet \
  --enable_saca \
  --saca_position after_stage0 \
  --train-root $IMAGES \
  --eval-root $IMAGES \
  --train-label $LABELS \
  --eval-label $LABELS \
  --train-list $TRAIN_LIST \
  --eval-list $EVAL_LIST \
  --seg-labels $SEG_LABELS \
  --image-ext .png \
  --label-suffix _label.npz \
  --label-mode 3 \
  --ce-class-weights "1,5,3,8" \
  --image-size 256 \
  --batch-size 32 \
  --epochs 60 \
  --plane axial \
  --vis-every 1 \
  --out-dir /kaggle/working/outputs \
  --run-name tumor_unet_saca_after_stage0
```

#### 13. UNet: Single SACA (Đặt ở lớp after_merge0)
```bash
!python -m swin_unet.src.cli train-tumor \
  --unet \
  --enable_saca \
  --saca_position after_merge0 \
  --train-root $IMAGES \
  --eval-root $IMAGES \
  --train-label $LABELS \
  --eval-label $LABELS \
  --train-list $TRAIN_LIST \
  --eval-list $EVAL_LIST \
  --seg-labels $SEG_LABELS \
  --image-ext .png \
  --label-suffix _label.npz \
  --label-mode 3 \
  --ce-class-weights "1,5,3,8" \
  --image-size 256 \
  --batch-size 32 \
  --epochs 60 \
  --plane axial \
  --vis-every 1 \
  --out-dir /kaggle/working/outputs \
  --run-name tumor_unet_saca_after_merge0
```

#### 14. UNet: Single SACA (Đặt ở lớp after_stage1)
```bash
!python -m swin_unet.src.cli train-tumor \
  --unet \
  --enable_saca \
  --saca_position after_stage1 \
  --train-root $IMAGES \
  --eval-root $IMAGES \
  --train-label $LABELS \
  --eval-label $LABELS \
  --train-list $TRAIN_LIST \
  --eval-list $EVAL_LIST \
  --seg-labels $SEG_LABELS \
  --image-ext .png \
  --label-suffix _label.npz \
  --label-mode 3 \
  --ce-class-weights "1,5,3,8" \
  --image-size 256 \
  --batch-size 32 \
  --epochs 60 \
  --plane axial \
  --vis-every 1 \
  --out-dir /kaggle/working/outputs \
  --run-name tumor_unet_saca_after_stage1
```

#### 15. UNet: Multi SACA (Đặt 2 lớp đầu)
Uses `after_patch_embed` and `after_stage0`.

```bash
!python -m swin_unet.src.cli train-tumor \
  --unet \
  --enable_saca \
  --saca_positions after_patch_embed,after_stage0 \
  --train-root $IMAGES \
  --eval-root $IMAGES \
  --train-label $LABELS \
  --eval-label $LABELS \
  --train-list $TRAIN_LIST \
  --eval-list $EVAL_LIST \
  --seg-labels $SEG_LABELS \
  --image-ext .png \
  --label-suffix _label.npz \
  --label-mode 3 \
  --ce-class-weights "1,5,3,8" \
  --image-size 256 \
  --batch-size 32 \
  --epochs 60 \
  --plane axial \
  --vis-every 1 \
  --out-dir /kaggle/working/outputs \
  --run-name tumor_unet_multi_saca_first_two
```

#### 16. UNet: Multi SACA (Đặt cả 4 lớp)
Uses `after_patch_embed`, `after_stage0`, `after_merge0`, and `after_stage1`.

```bash
!python -m swin_unet.src.cli train-tumor \
  --unet \
  --enable_saca \
  --saca_positions after_patch_embed,after_stage0,after_merge0,after_stage1 \
  --train-root $IMAGES \
  --eval-root $IMAGES \
  --train-label $LABELS \
  --eval-label $LABELS \
  --train-list $TRAIN_LIST \
  --eval-list $EVAL_LIST \
  --seg-labels $SEG_LABELS \
  --image-ext .png \
  --label-suffix _label.npz \
  --label-mode 3 \
  --ce-class-weights "1,5,3,8" \
  --image-size 256 \
  --batch-size 32 \
  --epochs 60 \
  --plane axial \
  --vis-every 1 \
  --out-dir /kaggle/working/outputs \
  --run-name tumor_unet_multi_saca_all_four
```


