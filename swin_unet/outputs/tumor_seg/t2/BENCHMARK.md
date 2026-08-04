# Bảng so sánh kết quả phân đoạn u não (T2 only)

Nguồn: `MaskReconstructContrastive` → `turmor_seg_t2/` (8 cấu hình × 30 epoch, modality **T2**).
Số **in đậm** = tốt nhất trong nhóm UNet (1–8).

---

## 1. Bảng chính — Region Dice (WT / TC / ET)

Metric BraTS chuẩn. **Epoch tốt nhất** chọn theo Dice TB = (WT+TC+ET)/3 trên eval.

### 1.1 Epoch tốt nhất

| Cấu hình | Epoch | WT Dice | WT IoU | WT HD95 | TC Dice | TC IoU | TC HD95 | ET Dice | ET IoU | ET HD95 | Dice TB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 17 | 0.891 | 0.803 | 7.64 | 0.752 | 0.603 | 5.68 | 0.559 | 0.388 | 5.45 | 0.734 |
| UNet: Dual View | 17 | 0.900 | 0.818 | **6.48** | 0.774 | 0.631 | 4.97 | 0.592 | 0.421 | 5.00 | 0.755 |
| UNet: SACA (after_patch_embed) | 19 | **0.902** | **0.822** | 6.65 | 0.772 | 0.628 | **4.62** | 0.595 | 0.424 | **4.52** | 0.756 |
| UNet: SACA (after_stage0) | 20 | 0.899 | 0.816 | 7.39 | 0.774 | 0.631 | 4.73 | 0.596 | 0.425 | 4.60 | 0.756 |
| UNet: SACA (after_merge0) | 14 | 0.901 | 0.820 | 6.58 | 0.781 | 0.640 | 4.84 | 0.596 | 0.424 | 4.76 | 0.759 |
| UNet: SACA (after_stage1) | 22 | 0.901 | 0.819 | 6.77 | 0.775 | 0.633 | 4.90 | 0.594 | 0.422 | 4.57 | 0.756 |
| UNet: Multi SACA (2 lớp) | 19 | 0.901 | 0.821 | 6.75 | 0.772 | 0.629 | 4.94 | 0.595 | 0.424 | 4.79 | 0.756 |
| UNet: Multi SACA (4 lớp) | 22 | 0.902 | 0.821 | 6.84 | **0.781** | **0.641** | 4.84 | **0.599** | **0.428** | 4.66 | **0.761** |

### 1.2 Epoch cuối cùng (epoch 30)

| Cấu hình | Epoch | WT Dice | WT IoU | WT HD95 | TC Dice | TC IoU | TC HD95 | ET Dice | ET IoU | ET HD95 | Dice TB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 30 | 0.890 | 0.802 | 8.38 | 0.740 | 0.588 | 5.78 | 0.549 | 0.378 | 5.44 | 0.726 |
| UNet: Dual View | 30 | 0.899 | 0.817 | 6.68 | 0.768 | 0.623 | 4.88 | 0.591 | 0.420 | 4.71 | 0.753 |
| UNet: SACA (after_patch_embed) | 30 | 0.901 | 0.820 | 6.69 | 0.772 | 0.628 | 4.78 | 0.587 | 0.415 | 4.58 | 0.753 |
| UNet: SACA (after_stage0) | 30 | 0.900 | 0.819 | 6.88 | 0.769 | 0.624 | **4.69** | 0.592 | 0.420 | 4.54 | 0.754 |
| UNet: SACA (after_merge0) | 30 | 0.901 | 0.821 | **6.60** | 0.774 | 0.631 | 4.85 | 0.591 | 0.420 | 4.66 | 0.756 |
| UNet: SACA (after_stage1) | 30 | 0.900 | 0.818 | 6.60 | 0.768 | 0.623 | 4.81 | 0.589 | 0.417 | **4.44** | 0.752 |
| UNet: Multi SACA (2 lớp) | 30 | **0.902** | **0.821** | 6.70 | 0.763 | 0.617 | 4.90 | 0.585 | 0.414 | 4.57 | 0.750 |
| UNet: Multi SACA (4 lớp) | 30 | 0.900 | 0.818 | 6.68 | **0.776** | **0.633** | 4.74 | **0.594** | **0.423** | 4.57 | **0.757** |

---

## 2. PC-Macro Dice & Loss (metric chọn checkpoint)

PC-Macro Dice = macro Dice trên các class có mặt, **loại Background**.

### 2.1 Epoch tốt nhất (theo PC-Macro)

| Cấu hình | Epoch tốt nhất | PC-Macro Dice | Macro Dice (all) | Eval Loss | Train PC-Macro | Gap (Train−Eval) |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 18 | 0.6133 | 0.6133 | 0.0965 | 0.8303 | 0.2170 |
| UNet: Dual View | 18 | 0.6426 | 0.6426 | 0.0865 | 0.8234 | 0.1809 |
| UNet: SACA (after_patch_embed) | 24 | 0.6344 | 0.6344 | 0.1095 | 0.8507 | 0.2163 |
| UNet: SACA (after_stage0) | 20 | 0.6296 | 0.6296 | 0.0934 | 0.8291 | 0.1995 |
| UNet: SACA (after_merge0) | 20 | 0.6502 | 0.6502 | 0.0929 | 0.8407 | 0.1906 |
| UNet: SACA (after_stage1) | 22 | 0.6490 | 0.6490 | 0.1024 | 0.8513 | 0.2023 |
| UNet: Multi SACA (2 lớp) | 18 | 0.6498 | 0.6498 | 0.0903 | 0.8326 | 0.1827 |
| UNet: Multi SACA (4 lớp) | 23 | 0.6470 | 0.6470 | 0.1009 | 0.8580 | 0.2110 |

### 2.2 Epoch cuối cùng

| Cấu hình | Epoch | PC-Macro Dice | Macro Dice (all) | Eval Loss | Train PC-Macro | Gap (Train−Eval) |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 30 | 0.6081 | 0.6081 | 0.1310 | 0.8716 | 0.2635 |
| UNet: Dual View | 30 | 0.6365 | 0.6365 | 0.1134 | 0.8582 | 0.2217 |
| UNet: SACA (after_patch_embed) | 30 | 0.6315 | 0.6315 | 0.1164 | 0.8590 | 0.2275 |
| UNet: SACA (after_stage0) | 30 | 0.6254 | 0.6254 | 0.1108 | 0.8528 | 0.2274 |
| UNet: SACA (after_merge0) | 30 | 0.6431 | 0.6431 | 0.1112 | 0.8647 | 0.2216 |
| UNet: SACA (after_stage1) | 30 | 0.6425 | 0.6425 | 0.1161 | 0.8667 | 0.2242 |
| UNet: Multi SACA (2 lớp) | 30 | 0.6358 | 0.6358 | 0.1131 | 0.8652 | 0.2294 |
| UNet: Multi SACA (4 lớp) | 30 | 0.6431 | 0.6431 | 0.1102 | 0.8702 | 0.2271 |

---

## 3. Per-class Dice trên eval

Dice từng lớp: Necrotic-Core · Edema · Enhancing-Tumor · Background.

### 3.1 Tại epoch tốt nhất (region Dice TB)

| Cấu hình | Epoch | Necrotic-Core | Edema | Enhancing-Tumor | Background | PC-Macro |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 17 | 0.5202 | 0.7565 | 0.5609 | 0.9979 | 0.6125 |
| UNet: Dual View | 17 | 0.4986 | 0.7766 | 0.5928 | 0.9980 | 0.6227 |
| UNet: SACA (after_patch_embed) | 19 | 0.4976 | 0.7812 | 0.5962 | 0.9981 | 0.6250 |
| UNet: SACA (after_stage0) | 20 | 0.5156 | 0.7767 | 0.5964 | 0.9980 | 0.6296 |
| UNet: SACA (after_merge0) | 14 | 0.5577 | 0.7777 | 0.5984 | 0.9980 | 0.6446 |
| UNet: SACA (after_stage1) | 22 | 0.5696 | 0.7814 | 0.5961 | 0.9980 | 0.6490 |
| UNet: Multi SACA (2 lớp) | 19 | 0.5331 | 0.7785 | 0.5965 | 0.9980 | 0.6360 |
| UNet: Multi SACA (4 lớp) | 22 | 0.5441 | 0.7832 | 0.6018 | 0.9981 | 0.6430 |

### 3.2 Tại epoch cuối cùng

| Cấu hình | Epoch | Necrotic-Core | Edema | Enhancing-Tumor | Background | PC-Macro |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 30 | 0.5133 | 0.7596 | 0.5514 | 0.9978 | 0.6081 |
| UNet: Dual View | 30 | 0.5370 | 0.7794 | 0.5933 | 0.9980 | 0.6365 |
| UNet: SACA (after_patch_embed) | 30 | 0.5257 | 0.7810 | 0.5878 | 0.9980 | 0.6315 |
| UNet: SACA (after_stage0) | 30 | 0.5051 | 0.7795 | 0.5916 | 0.9980 | 0.6254 |
| UNet: SACA (after_merge0) | 30 | 0.5538 | 0.7834 | 0.5923 | 0.9980 | 0.6431 |
| UNet: SACA (after_stage1) | 30 | 0.5576 | 0.7796 | 0.5903 | 0.9980 | 0.6425 |
| UNet: Multi SACA (2 lớp) | 30 | 0.5415 | 0.7799 | 0.5861 | 0.9981 | 0.6358 |
| UNet: Multi SACA (4 lớp) | 30 | 0.5496 | 0.7816 | 0.5980 | 0.9980 | 0.6431 |

---

## 4. Suy giảm: Best epoch → Epoch 30

Δ càng nhỏ càng ổn định (ít overfit / ít mất hiệu năng cuối training).

| Cấu hình | Best Ep (Reg) | Dice TB best | Dice TB last | Δ Dice TB | Best Ep (PC) | PC-Macro best | PC-Macro last | Δ PC-Macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 17 | 0.734 | 0.726 | -0.0077 | 18 | 0.6133 | 0.6081 | -0.0052 |
| UNet: Dual View | 17 | 0.755 | 0.753 | -0.0027 | 18 | 0.6426 | 0.6365 | -0.0060 |
| UNet: SACA (after_patch_embed) | 19 | 0.756 | 0.753 | -0.0033 | 24 | 0.6344 | 0.6315 | -0.0028 |
| UNet: SACA (after_stage0) | 20 | 0.756 | 0.754 | -0.0026 | 20 | 0.6296 | 0.6254 | -0.0042 |
| UNet: SACA (after_merge0) | 14 | 0.759 | 0.756 | -0.0036 | 20 | 0.6502 | 0.6431 | -0.0070 |
| UNet: SACA (after_stage1) | 22 | 0.756 | 0.752 | -0.0042 | 22 | 0.6490 | 0.6425 | -0.0065 |
| UNet: Multi SACA (2 lớp) | 19 | 0.756 | 0.750 | -0.0062 | 18 | 0.6498 | 0.6358 | -0.0140 |
| UNet: Multi SACA (4 lớp) | 22 | 0.761 | 0.757 | -0.0038 | 23 | 0.6470 | 0.6431 | -0.0039 |

---

## 5. Generalization — Train vs Eval (tại epoch tốt nhất region)

Gap = Train − Eval. Gap lớn → model fit train mạnh hơn eval.

| Cấu hình | Epoch | Train Dice TB | Eval Dice TB | Gap Reg | Train PC-Macro | Eval PC-Macro | Gap PC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 17 | 0.872 | 0.734 | 0.1380 | 0.8226 | 0.6125 | 0.2101 |
| UNet: Dual View | 17 | 0.868 | 0.755 | 0.1131 | 0.8164 | 0.6227 | 0.1937 |
| UNet: SACA (after_patch_embed) | 19 | 0.879 | 0.756 | 0.1221 | 0.8294 | 0.6250 | 0.2044 |
| UNet: SACA (after_stage0) | 20 | 0.878 | 0.756 | 0.1213 | 0.8291 | 0.6296 | 0.1995 |
| UNet: SACA (after_merge0) | 14 | 0.859 | 0.759 | 0.0996 | 0.8048 | 0.6446 | 0.1602 |
| UNet: SACA (after_stage1) | 22 | 0.895 | 0.756 | 0.1382 | 0.8513 | 0.6490 | 0.2023 |
| UNet: Multi SACA (2 lớp) | 19 | 0.884 | 0.756 | 0.1273 | 0.8372 | 0.6360 | 0.2012 |
| UNet: Multi SACA (4 lớp) | 22 | 0.897 | 0.761 | 0.1369 | 0.8555 | 0.6430 | 0.2125 |

---

## Đánh giá tổng hợp

- **Tốt nhất theo region (Dice TB)**: **UNet: Multi SACA (4 lớp)** (epoch 22) — Dice TB **0.761**, WT 0.902 / TC 0.781 / ET 0.599.
- **Tốt nhất theo PC-Macro Dice**: **UNet: SACA (after_merge0)** (epoch 20) — PC-Macro **0.6502**.

---

## Nguồn dữ liệu

Mỗi thư mục `turmor_seg_t2/<exp>/`:

- `region_dice_log.csv` — WT/TC/ET Dice, IoU, HD95 (train & eval)
- `epoch_log.csv` — loss, macro Dice, PC-Macro Dice
- `per_class_dice_eval.csv` — Dice từng class trên eval
- `reports/epoch_reports.jsonl` — log JSON theo epoch
