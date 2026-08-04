# Benchmark — Tumor Segmentation (Stacked Channels & Modality Dropout)

Nguồn: `MaskReconstructContrastive` → `turmor_seg_stack/` (5 cấu hình × 60 epoch, modality **T1 + T1CE + T2 + FLAIR** / **+ CER + HYPER** (`--stack-modality-channels`)).
Số **in đậm** = tốt nhất trong nhóm (5 cấu hình UNet).
Quy tắc: Dice / IoU **cao hơn = tốt hơn** · HD95 **thấp hơn = tốt hơn**.

---

## 1. Region Dice (WT / TC / ET)

Metric BraTS chuẩn. **Epoch tốt nhất** chọn theo Dice TB = (WT+TC+ET)/3 trên eval.

### 1.1 Epoch tốt nhất

| Cấu hình | Epoch | WT Dice | WT IoU | WT HD95 | TC Dice | TC IoU | TC HD95 | ET Dice | ET IoU | ET HD95 | Dice TB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: SACA (after_patch_embed, stack soft) | 46 | 0.937 | 0.881 | 5.265 | 0.884 | 0.791 | 2.586 | 0.884 | 0.793 | 2.059 | 0.902 |
| UNet: Multi SACA (4 lớp, stack) | 17 | 0.933 | 0.875 | 5.596 | **0.893** | **0.806** | 2.809 | 0.875 | 0.778 | 2.358 | 0.900 |
| UNet: Multi SACA (4 lớp, stack soft) | 42 | 0.937 | 0.881 | 5.500 | 0.883 | 0.791 | 2.578 | **0.887** | **0.797** | 2.042 | **0.903** |
| UNet: SACA (after_patch_embed, stack d10) | 55 | 0.937 | 0.881 | 4.990 | 0.879 | 0.784 | **2.379** | 0.886 | 0.795 | **1.962** | 0.900 |
| UNet: Multi SACA (4 lớp, stack d10) | 47 | **0.937** | **0.882** | **4.862** | 0.886 | 0.795 | 2.454 | 0.884 | 0.792 | 1.973 | 0.902 |

### 1.2 Epoch cuối cùng (epoch 60)

| Cấu hình | Epoch | WT Dice | WT IoU | WT HD95 | TC Dice | TC IoU | TC HD95 | ET Dice | ET IoU | ET HD95 | Dice TB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: SACA (after_patch_embed, stack soft) | 60 | 0.937 | 0.881 | 5.284 | 0.881 | 0.787 | 2.624 | 0.884 | 0.793 | 2.079 | 0.901 |
| UNet: Multi SACA (4 lớp, stack) | 60 | 0.933 | 0.875 | 5.829 | 0.874 | 0.776 | 2.625 | 0.884 | 0.792 | 2.077 | 0.897 |
| UNet: Multi SACA (4 lớp, stack soft) | 60 | **0.937** | **0.881** | 5.312 | 0.876 | 0.780 | 2.680 | **0.886** | **0.796** | 2.089 | 0.900 |
| UNet: SACA (after_patch_embed, stack d10) | 60 | 0.937 | 0.881 | 5.097 | 0.880 | 0.785 | **2.392** | 0.884 | 0.792 | 1.984 | 0.900 |
| UNet: Multi SACA (4 lớp, stack d10) | 60 | 0.936 | 0.880 | **4.869** | **0.885** | **0.794** | 2.441 | 0.885 | 0.794 | **1.968** | **0.902** |

---

## 2. PC-Macro Dice & Loss (metric chọn checkpoint)

PC-Macro Dice = macro Dice trên các class có mặt, **loại Background**.

### 2.1 Epoch tốt nhất (theo PC-Macro)

| Cấu hình | Epoch tốt nhất | PC-Macro Dice | Macro Dice (all) | Eval Loss | Train PC-Macro | Gap (Train−Eval) |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: SACA (after_patch_embed, stack soft) | 21 | 0.8320 | 0.8320 | 0.0273 | 0.9210 | 0.0890 |
| UNet: Multi SACA (4 lớp, stack) | 17 | **0.8397** | 0.8397 | 0.0467 | 0.9096 | 0.0699 |
| UNet: Multi SACA (4 lớp, stack soft) | 17 | 0.8354 | 0.8354 | 0.0270 | 0.9152 | 0.0798 |
| UNet: SACA (after_patch_embed, stack d10) | 19 | 0.8326 | 0.8326 | 0.0357 | 0.8951 | 0.0626 |
| UNet: Multi SACA (4 lớp, stack d10) | 14 | 0.8345 | 0.8345 | 0.0404 | 0.8757 | 0.0412 |

### 2.2 Epoch cuối cùng

| Cấu hình | Epoch | PC-Macro Dice | Macro Dice (all) | Eval Loss | Train PC-Macro | Gap (Train−Eval) |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: SACA (after_patch_embed, stack soft) | 60 | 0.8293 | 0.8293 | 0.0381 | 0.9487 | 0.1194 |
| UNet: Multi SACA (4 lớp, stack) | 60 | 0.8194 | 0.8194 | 0.0788 | 0.9493 | 0.1299 |
| UNet: Multi SACA (4 lớp, stack soft) | 60 | 0.8251 | 0.8251 | 0.0400 | 0.9524 | 0.1273 |
| UNet: SACA (after_patch_embed, stack d10) | 60 | 0.8255 | 0.8255 | 0.0564 | 0.9328 | 0.1073 |
| UNet: Multi SACA (4 lớp, stack d10) | 60 | **0.8314** | 0.8314 | 0.0528 | 0.9312 | 0.0998 |

---

## 3. Per-class Dice trên eval

Dice từng lớp: Necrotic-Core · Edema · Enhancing-Tumor · Background.

### 3.1 Tại epoch tốt nhất (region Dice TB)

| Cấu hình | Epoch | Necrotic-Core | Edema | Enhancing-Tumor | Background | PC-Macro |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: SACA (after_patch_embed, stack soft) | 46 | 0.7436 | 0.8683 | 0.8841 | 0.9988 | 0.8320 |
| UNet: Multi SACA (4 lớp, stack) | 17 | **0.7789** | 0.8652 | 0.8751 | 0.9987 | **0.8397** |
| UNet: Multi SACA (4 lớp, stack soft) | 42 | 0.7427 | 0.8690 | **0.8869** | 0.9988 | 0.8329 |
| UNet: SACA (after_patch_embed, stack d10) | 55 | 0.7226 | 0.8668 | 0.8851 | **0.9988** | 0.8249 |
| UNet: Multi SACA (4 lớp, stack d10) | 47 | 0.7450 | **0.8694** | 0.8835 | **0.9988** | 0.8326 |

### 3.2 Tại epoch cuối cùng

| Cấu hình | Epoch | Necrotic-Core | Edema | Enhancing-Tumor | Background | PC-Macro |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: SACA (after_patch_embed, stack soft) | 60 | 0.7363 | 0.8673 | 0.8842 | 0.9988 | 0.8293 |
| UNet: Multi SACA (4 lớp, stack) | 60 | 0.7163 | 0.8581 | 0.8838 | 0.9987 | 0.8194 |
| UNet: Multi SACA (4 lớp, stack soft) | 60 | 0.7231 | 0.8666 | **0.8857** | 0.9988 | 0.8251 |
| UNet: SACA (after_patch_embed, stack d10) | 60 | 0.7259 | 0.8670 | 0.8837 | **0.9988** | 0.8255 |
| UNet: Multi SACA (4 lớp, stack d10) | 60 | **0.7409** | **0.8681** | 0.8852 | 0.9988 | **0.8314** |

---

## 4. Suy giảm: Best epoch → Epoch 60

Δ càng nhỏ càng ổn định (ít overfit / ít mất hiệu năng cuối training).

| Cấu hình | Best Ep (Reg) | Dice TB best | Dice TB last | Δ Dice TB | Best Ep (PC) | PC-Macro best | PC-Macro last | Δ PC-Macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: SACA (after_patch_embed, stack soft) | 46 | 0.902 | 0.901 | -0.0009 | 21 | 0.8320 | 0.8293 | -0.0027 |
| UNet: Multi SACA (4 lớp, stack) | 17 | 0.900 | 0.897 | -0.0033 | 17 | 0.8397 | 0.8194 | -0.0203 |
| UNet: Multi SACA (4 lớp, stack soft) | 42 | 0.903 | 0.900 | -0.0028 | 17 | 0.8354 | 0.8251 | -0.0103 |
| UNet: SACA (after_patch_embed, stack d10) | 55 | 0.900 | 0.900 | -0.0003 | 19 | 0.8326 | 0.8255 | -0.0070 |
| UNet: Multi SACA (4 lớp, stack d10) | 47 | 0.902 | 0.902 | -0.0001 | 14 | 0.8345 | 0.8314 | -0.0031 |

---

## Đánh giá tổng hợp

- **Tốt nhất theo region (Dice TB)**: **UNet: Multi SACA (4 lớp, stack soft)** (epoch 42) — Dice TB **0.903**, WT 0.937 / TC 0.883 / ET 0.887 (HD95 TC 2.578, HD95 ET 2.042).
- **Tốt nhất theo PC-Macro Dice**: **UNet: Multi SACA (4 lớp, stack)** (epoch 17) — PC-Macro **0.8397**.
- **Tốt nhất UNet + SACA**: **UNet: Multi SACA (4 lớp, stack soft)** (epoch 42) — Dice TB **0.903** & PC-Macro **0.8397**.

---

## Nguồn dữ liệu

Mỗi thư mục `turmor_seg_stack/<exp>/`:

- `region_dice_log.csv` — WT/TC/ET Dice, IoU, HD95 (train & eval)
- `epoch_log.csv` — loss, macro Dice, PC-Macro Dice
- `per_class_dice_eval.csv` — Dice từng class trên eval
- `reports/epoch_reports.jsonl` — log JSON theo epoch
