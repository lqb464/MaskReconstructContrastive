# Bảng so sánh kết quả phân đoạn u não (T1CE only)

Nguồn: `MaskReconstructContrastive` → `turmor_seg_t1ce/` (8 cấu hình × 30 epoch, modality **T1CE**).
Số **in đậm** = tốt nhất trong nhóm UNet (1–8).

---

## 1. Bảng chính — Region Dice (WT / TC / ET)

Metric BraTS chuẩn. **Epoch tốt nhất** chọn theo Dice TB = (WT+TC+ET)/3 trên eval.

### 1.1 Epoch tốt nhất

| Cấu hình | Epoch | WT Dice | WT IoU | WT HD95 | TC Dice | TC IoU | TC HD95 | ET Dice | ET IoU | ET HD95 | Dice TB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 21 | 0.828 | 0.707 | 10.59 | 0.858 | 0.752 | 3.30 | 0.860 | 0.754 | 2.63 | 0.849 |
| UNet: Dual View | 23 | 0.841 | 0.725 | 9.62 | 0.860 | 0.755 | 3.16 | 0.856 | 0.749 | 2.50 | 0.852 |
| UNet: SACA (after_patch_embed) | 25 | 0.844 | 0.731 | 9.38 | **0.870** | **0.770** | **2.83** | 0.869 | 0.769 | 2.35 | **0.861** |
| UNet: SACA (after_stage0) | 25 | 0.844 | 0.730 | 9.54 | 0.866 | 0.763 | 3.12 | 0.868 | 0.766 | 2.41 | 0.859 |
| UNet: SACA (after_merge0) | 16 | 0.841 | 0.726 | 9.62 | 0.870 | 0.770 | 3.16 | 0.867 | 0.766 | 2.52 | 0.859 |
| UNet: SACA (after_stage1) | 23 | 0.842 | 0.728 | **9.25** | 0.864 | 0.761 | 2.95 | 0.864 | 0.760 | **2.31** | 0.857 |
| UNet: Multi SACA (2 lớp) | 25 | **0.846** | **0.734** | 9.27 | 0.869 | 0.768 | 3.16 | 0.867 | 0.765 | 2.53 | 0.861 |
| UNet: Multi SACA (4 lớp) | 25 | 0.843 | 0.729 | 9.43 | 0.865 | 0.762 | 3.03 | **0.871** | **0.771** | 2.33 | 0.859 |

### 1.2 Epoch cuối cùng (epoch 30)

| Cấu hình | Epoch | WT Dice | WT IoU | WT HD95 | TC Dice | TC IoU | TC HD95 | ET Dice | ET IoU | ET HD95 | Dice TB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 30 | 0.826 | 0.704 | 11.05 | 0.854 | 0.745 | 3.49 | 0.859 | 0.753 | 2.77 | 0.846 |
| UNet: Dual View | 30 | 0.841 | 0.726 | 9.49 | 0.857 | 0.750 | 3.17 | 0.854 | 0.745 | 2.53 | 0.851 |
| UNet: SACA (after_patch_embed) | 30 | 0.845 | 0.731 | 9.34 | **0.869** | **0.768** | **2.86** | 0.868 | 0.766 | 2.36 | **0.860** |
| UNet: SACA (after_stage0) | 30 | 0.844 | 0.730 | 9.44 | 0.864 | 0.761 | 3.13 | 0.866 | 0.764 | 2.49 | 0.858 |
| UNet: SACA (after_merge0) | 30 | 0.842 | 0.727 | 9.50 | 0.866 | 0.764 | 3.12 | 0.866 | 0.764 | 2.45 | 0.858 |
| UNet: SACA (after_stage1) | 30 | 0.842 | 0.727 | 9.25 | 0.863 | 0.760 | 2.94 | 0.862 | 0.757 | **2.29** | 0.856 |
| UNet: Multi SACA (2 lớp) | 30 | **0.847** | **0.734** | **9.14** | 0.867 | 0.765 | 3.07 | 0.867 | 0.765 | 2.40 | 0.860 |
| UNet: Multi SACA (4 lớp) | 30 | 0.842 | 0.727 | 9.43 | 0.862 | 0.758 | 3.05 | **0.869** | **0.768** | 2.31 | 0.858 |

---

## 2. PC-Macro Dice & Loss (metric chọn checkpoint)

PC-Macro Dice = macro Dice trên các class có mặt, **loại Background**.

### 2.1 Epoch tốt nhất (theo PC-Macro)

| Cấu hình | Epoch tốt nhất | PC-Macro Dice | Macro Dice (all) | Eval Loss | Train PC-Macro | Gap (Train−Eval) |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 7 | 0.7639 | 0.7639 | 0.0724 | 0.8158 | 0.0519 |
| UNet: Dual View | 23 | 0.7635 | 0.7635 | 0.0941 | 0.8915 | 0.1280 |
| UNet: SACA (after_patch_embed) | 23 | 0.7785 | 0.7785 | 0.0871 | 0.8976 | 0.1192 |
| UNet: SACA (after_stage0) | 14 | 0.7771 | 0.7771 | 0.0727 | 0.8679 | 0.0908 |
| UNet: SACA (after_merge0) | 7 | 0.7759 | 0.7759 | 0.0621 | 0.8205 | 0.0446 |
| UNet: SACA (after_stage1) | 19 | 0.7696 | 0.7696 | 0.0814 | 0.8922 | 0.1226 |
| UNet: Multi SACA (2 lớp) | 18 | 0.7794 | 0.7794 | 0.0677 | 0.8837 | 0.1042 |
| UNet: Multi SACA (4 lớp) | 14 | 0.7750 | 0.7750 | 0.0717 | 0.8727 | 0.0977 |

### 2.2 Epoch cuối cùng

| Cấu hình | Epoch | PC-Macro Dice | Macro Dice (all) | Eval Loss | Train PC-Macro | Gap (Train−Eval) |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 30 | 0.7554 | 0.7554 | 0.1180 | 0.9127 | 0.1573 |
| UNet: Dual View | 30 | 0.7588 | 0.7588 | 0.1033 | 0.9001 | 0.1413 |
| UNet: SACA (after_patch_embed) | 30 | 0.7758 | 0.7758 | 0.0967 | 0.9056 | 0.1298 |
| UNet: SACA (after_stage0) | 30 | 0.7713 | 0.7713 | 0.0980 | 0.9068 | 0.1356 |
| UNet: SACA (after_merge0) | 30 | 0.7695 | 0.7695 | 0.1027 | 0.9088 | 0.1393 |
| UNet: SACA (after_stage1) | 30 | 0.7671 | 0.7671 | 0.1039 | 0.9121 | 0.1451 |
| UNet: Multi SACA (2 lớp) | 30 | 0.7745 | 0.7745 | 0.0918 | 0.9058 | 0.1313 |
| UNet: Multi SACA (4 lớp) | 30 | 0.7696 | 0.7696 | 0.1030 | 0.9122 | 0.1426 |

---

## 3. Per-class Dice trên eval

Dice từng lớp: Necrotic-Core · Edema · Enhancing-Tumor · Background.

### 3.1 Tại epoch tốt nhất (region Dice TB)

| Cấu hình | Epoch | Necrotic-Core | Edema | Enhancing-Tumor | Background | PC-Macro |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 21 | 0.7024 | 0.7140 | 0.8602 | 0.9966 | 0.7589 |
| UNet: Dual View | 23 | 0.7064 | 0.7280 | 0.8561 | 0.9968 | 0.7635 |
| UNet: SACA (after_patch_embed) | 25 | 0.7288 | 0.7353 | 0.8690 | 0.9969 | 0.7777 |
| UNet: SACA (after_stage0) | 25 | 0.7184 | 0.7358 | 0.8672 | 0.9969 | 0.7738 |
| UNet: SACA (after_merge0) | 16 | 0.7259 | 0.7306 | 0.8670 | 0.9969 | 0.7745 |
| UNet: SACA (after_stage1) | 23 | 0.7109 | 0.7337 | 0.8638 | 0.9970 | 0.7695 |
| UNet: Multi SACA (2 lớp) | 25 | 0.7273 | 0.7378 | 0.8665 | 0.9970 | 0.7772 |
| UNet: Multi SACA (4 lớp) | 25 | 0.7137 | 0.7336 | 0.8712 | 0.9970 | 0.7728 |

### 3.2 Tại epoch cuối cùng

| Cấu hình | Epoch | Necrotic-Core | Edema | Enhancing-Tumor | Background | PC-Macro |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 30 | 0.6970 | 0.7101 | 0.8590 | 0.9966 | 0.7554 |
| UNet: Dual View | 30 | 0.6945 | 0.7279 | 0.8539 | 0.9969 | 0.7588 |
| UNet: SACA (after_patch_embed) | 30 | 0.7244 | 0.7352 | 0.8678 | 0.9970 | 0.7758 |
| UNet: SACA (after_stage0) | 30 | 0.7124 | 0.7358 | 0.8656 | 0.9970 | 0.7713 |
| UNet: SACA (after_merge0) | 30 | 0.7114 | 0.7316 | 0.8655 | 0.9969 | 0.7695 |
| UNet: SACA (after_stage1) | 30 | 0.7071 | 0.7318 | 0.8624 | 0.9969 | 0.7671 |
| UNet: Multi SACA (2 lớp) | 30 | 0.7189 | 0.7378 | 0.8669 | 0.9970 | 0.7745 |
| UNet: Multi SACA (4 lớp) | 30 | 0.7076 | 0.7316 | 0.8695 | 0.9969 | 0.7696 |

---

## 4. Suy giảm: Best epoch → Epoch 30

Δ càng nhỏ càng ổn định (ít overfit / ít mất hiệu năng cuối training).

| Cấu hình | Best Ep (Reg) | Dice TB best | Dice TB last | Δ Dice TB | Best Ep (PC) | PC-Macro best | PC-Macro last | Δ PC-Macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 21 | 0.849 | 0.846 | -0.0024 | 7 | 0.7639 | 0.7554 | -0.0086 |
| UNet: Dual View | 23 | 0.852 | 0.851 | -0.0016 | 23 | 0.7635 | 0.7588 | -0.0047 |
| UNet: SACA (after_patch_embed) | 25 | 0.861 | 0.860 | -0.0009 | 23 | 0.7785 | 0.7758 | -0.0027 |
| UNet: SACA (after_stage0) | 25 | 0.859 | 0.858 | -0.0009 | 14 | 0.7771 | 0.7713 | -0.0058 |
| UNet: SACA (after_merge0) | 16 | 0.859 | 0.858 | -0.0014 | 7 | 0.7759 | 0.7695 | -0.0064 |
| UNet: SACA (after_stage1) | 23 | 0.857 | 0.856 | -0.0010 | 19 | 0.7696 | 0.7671 | -0.0025 |
| UNet: Multi SACA (2 lớp) | 25 | 0.861 | 0.860 | -0.0007 | 18 | 0.7794 | 0.7745 | -0.0049 |
| UNet: Multi SACA (4 lớp) | 25 | 0.859 | 0.858 | -0.0018 | 14 | 0.7750 | 0.7696 | -0.0054 |

---

## 5. Generalization — Train vs Eval (tại epoch tốt nhất region)

Gap = Train − Eval. Gap lớn → model fit train mạnh hơn eval.

| Cấu hình | Epoch | Train Dice TB | Eval Dice TB | Gap Reg | Train PC-Macro | Eval PC-Macro | Gap PC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 21 | 0.926 | 0.849 | 0.0769 | 0.8984 | 0.7589 | 0.1395 |
| UNet: Dual View | 23 | 0.921 | 0.852 | 0.0685 | 0.8915 | 0.7635 | 0.1280 |
| UNet: SACA (after_patch_embed) | 25 | 0.928 | 0.861 | 0.0672 | 0.9021 | 0.7777 | 0.1244 |
| UNet: SACA (after_stage0) | 25 | 0.929 | 0.859 | 0.0701 | 0.9033 | 0.7738 | 0.1295 |
| UNet: SACA (after_merge0) | 16 | 0.911 | 0.859 | 0.0520 | 0.8790 | 0.7745 | 0.1045 |
| UNet: SACA (after_stage1) | 23 | 0.930 | 0.857 | 0.0734 | 0.9040 | 0.7695 | 0.1346 |
| UNet: Multi SACA (2 lớp) | 25 | 0.928 | 0.861 | 0.0677 | 0.9025 | 0.7772 | 0.1253 |
| UNet: Multi SACA (4 lớp) | 25 | 0.932 | 0.859 | 0.0730 | 0.9086 | 0.7728 | 0.1358 |

---

## Đánh giá tổng hợp

- **Tốt nhất theo region (Dice TB)**: **UNet: SACA (after_patch_embed)** (epoch 25) — Dice TB **0.861**, WT 0.844 / TC 0.870 / ET 0.869.
- **Tốt nhất theo PC-Macro Dice**: **UNet: Multi SACA (2 lớp)** (epoch 18) — PC-Macro **0.7794**.

---

## Nguồn dữ liệu

Mỗi thư mục `turmor_seg_t1ce/<exp>/`:

- `region_dice_log.csv` — WT/TC/ET Dice, IoU, HD95 (train & eval)
- `epoch_log.csv` — loss, macro Dice, PC-Macro Dice
- `per_class_dice_eval.csv` — Dice từng class trên eval
- `reports/epoch_reports.jsonl` — log JSON theo epoch
