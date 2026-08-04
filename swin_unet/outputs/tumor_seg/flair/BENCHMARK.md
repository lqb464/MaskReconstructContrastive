# Benchmark — Tumor Segmentation (FLAIR only)

Nguồn: `MaskReconstructContrastive` → `turmor_seg_flair/` (12 cấu hình × 60 epoch, modality **FLAIR**).
Số **in đậm** = tốt nhất trong nhóm UNet (1–10) hoặc Swin-UNet (11–12).
Quy tắc: Dice / IoU **cao hơn = tốt hơn** · HD95 **thấp hơn = tốt hơn**.

---

## 1. Region Dice (WT / TC / ET)

Metric BraTS chuẩn. **Epoch tốt nhất** chọn theo Dice TB = (WT+TC+ET)/3 trên eval.

### 1.1 Epoch tốt nhất

| Cấu hình | Epoch | WT Dice | WT IoU | WT HD95 | TC Dice | TC IoU | TC HD95 | ET Dice | ET IoU | ET HD95 | Dice TB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 21 | 0.903 | 0.824 | 7.509 | 0.733 | 0.579 | 5.753 | 0.536 | 0.366 | 5.691 | 0.724 |
| UNet: Dual View | 20 | 0.919 | 0.851 | 6.088 | 0.760 | 0.613 | 4.805 | 0.571 | 0.399 | **4.940** | 0.750 |
| UNet: SACA (after_patch_embed) | 27 | **0.923** | **0.858** | 5.765 | 0.771 | 0.627 | 4.872 | **0.571** | **0.400** | 4.972 | **0.755** |
| UNet: SACA (after_patch_embed, soft) | 27 | 0.914 | 0.842 | 5.706 | 0.752 | 0.603 | 4.956 | 0.550 | 0.379 | 5.024 | 0.739 |
| UNet: SACA (after_stage0) | 22 | 0.917 | 0.847 | **5.647** | 0.751 | 0.601 | 5.135 | 0.552 | 0.381 | 5.038 | 0.740 |
| UNet: SACA (after_stage0, soft) | 21 | 0.916 | 0.845 | 6.469 | 0.765 | 0.619 | 5.003 | 0.567 | 0.396 | 5.183 | 0.749 |
| UNet: SACA (after_merge0) | 12 | 0.921 | 0.853 | 5.709 | 0.766 | 0.621 | 4.829 | 0.567 | 0.396 | 5.292 | 0.751 |
| UNet: SACA (after_stage1) | 29 | 0.904 | 0.825 | 6.162 | 0.764 | 0.618 | 5.030 | 0.555 | 0.384 | 5.119 | 0.741 |
| UNet: Multi SACA (2 lớp) | 31 | 0.921 | 0.854 | 6.010 | 0.768 | 0.624 | 4.977 | 0.567 | 0.396 | 5.129 | 0.752 |
| UNet: Multi SACA (4 lớp) | 12 | 0.917 | 0.846 | 6.082 | **0.772** | **0.629** | **4.777** | 0.567 | 0.396 | 5.163 | 0.752 |
| Swin-UNet: Single View | 16 | **0.879** | **0.784** | 8.638 | 0.562 | 0.391 | 9.360 | 0.359 | 0.219 | 9.101 | 0.600 |
| Swin-UNet: Dual View | 13 | 0.872 | 0.774 | **8.322** | **0.577** | **0.405** | **8.069** | **0.383** | **0.237** | **7.795** | **0.611** |

### 1.2 Epoch cuối cùng (epoch 60)

| Cấu hình | Epoch | WT Dice | WT IoU | WT HD95 | TC Dice | TC IoU | TC HD95 | ET Dice | ET IoU | ET HD95 | Dice TB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 60 | 0.907 | 0.829 | 7.221 | 0.724 | 0.567 | 6.255 | 0.519 | 0.350 | 5.865 | 0.716 |
| UNet: Dual View | 60 | 0.914 | 0.841 | 5.940 | 0.759 | 0.611 | 4.916 | 0.549 | 0.379 | 4.927 | 0.741 |
| UNet: SACA (after_patch_embed) | 60 | 0.915 | 0.844 | 6.493 | **0.762** | **0.615** | 4.959 | **0.556** | **0.385** | 4.911 | **0.744** |
| UNet: SACA (after_patch_embed, soft) | 60 | 0.910 | 0.834 | 5.793 | 0.732 | 0.578 | 4.991 | 0.534 | 0.365 | 4.884 | 0.725 |
| UNet: SACA (after_stage0) | 60 | 0.913 | 0.841 | 5.749 | 0.741 | 0.589 | 5.055 | 0.537 | 0.367 | 4.803 | 0.730 |
| UNet: SACA (after_stage0, soft) | 60 | 0.919 | 0.851 | 5.890 | 0.755 | 0.606 | 4.977 | 0.549 | 0.378 | 4.994 | 0.741 |
| UNet: SACA (after_merge0) | 60 | **0.921** | **0.854** | **5.667** | 0.753 | 0.604 | 4.990 | 0.546 | 0.376 | 4.969 | 0.740 |
| UNet: SACA (after_stage1) | 60 | 0.905 | 0.826 | 5.886 | 0.751 | 0.602 | 4.876 | 0.540 | 0.370 | 4.855 | 0.732 |
| UNet: Multi SACA (2 lớp) | 60 | 0.915 | 0.844 | 6.245 | 0.759 | 0.612 | **4.792** | 0.554 | 0.383 | **4.781** | 0.743 |
| UNet: Multi SACA (4 lớp) | 60 | 0.902 | 0.822 | 5.948 | 0.753 | 0.604 | 5.025 | 0.548 | 0.377 | 4.965 | 0.734 |
| Swin-UNet: Single View | 60 | 0.878 | 0.782 | 8.124 | **0.515** | **0.347** | 9.036 | 0.330 | 0.197 | 8.371 | 0.574 |
| Swin-UNet: Dual View | 60 | **0.882** | **0.789** | **7.111** | 0.512 | 0.344 | **8.243** | **0.339** | **0.204** | **7.646** | **0.578** |

---

## 2. PC-Macro Dice & Loss (metric chọn checkpoint)

PC-Macro Dice = macro Dice trên các class có mặt, **loại Background**.

### 2.1 Epoch tốt nhất (theo PC-Macro)

| Cấu hình | Epoch tốt nhất | PC-Macro Dice | Macro Dice (all) | Eval Loss | Train PC-Macro | Gap (Train−Eval) |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 35 | 0.5706 | 0.5706 | 0.1343 | 0.8804 | 0.3097 |
| UNet: Dual View | 15 | 0.6068 | 0.6068 | 0.0968 | 0.7887 | 0.1819 |
| UNet: SACA (after_patch_embed) | 12 | 0.6187 | 0.6187 | 0.0835 | 0.7744 | 0.1556 |
| UNet: SACA (after_patch_embed, soft) | 19 | 0.5961 | 0.5961 | 0.0519 | 0.8234 | 0.2273 |
| UNet: SACA (after_stage0) | 14 | 0.6093 | 0.6093 | 0.1011 | 0.7962 | 0.1869 |
| UNet: SACA (after_stage0, soft) | 42 | 0.6028 | 0.6028 | 0.0663 | 0.8868 | 0.2840 |
| UNet: SACA (after_merge0) | 16 | 0.6082 | 0.6082 | 0.0940 | 0.8068 | 0.1985 |
| UNet: SACA (after_stage1) | 29 | 0.5981 | 0.5981 | 0.1090 | 0.8654 | 0.2673 |
| UNet: Multi SACA (2 lớp) | 30 | 0.6090 | 0.6090 | 0.1033 | 0.8573 | 0.2483 |
| UNet: Multi SACA (4 lớp) | 17 | 0.6101 | 0.6101 | 0.0955 | 0.8142 | 0.2041 |
| Swin-UNet: Single View | 11 | 0.4154 | 0.4154 | 0.1330 | 0.6725 | 0.2571 |
| Swin-UNet: Dual View | 12 | 0.4301 | 0.4301 | 0.1401 | 0.7330 | 0.3029 |

### 2.2 Epoch cuối cùng

| Cấu hình | Epoch | PC-Macro Dice | Macro Dice (all) | Eval Loss | Train PC-Macro | Gap (Train−Eval) |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 60 | 0.5628 | 0.5628 | 0.1776 | 0.9103 | 0.3475 |
| UNet: Dual View | 60 | 0.5897 | 0.5897 | 0.1500 | 0.8942 | 0.3045 |
| UNet: SACA (after_patch_embed) | 60 | 0.6041 | 0.6041 | 0.1405 | 0.8980 | 0.2939 |
| UNet: SACA (after_patch_embed, soft) | 60 | 0.5841 | 0.5841 | 0.0732 | 0.8981 | 0.3140 |
| UNet: SACA (after_stage0) | 60 | 0.6019 | 0.6019 | 0.1530 | 0.9001 | 0.2982 |
| UNet: SACA (after_stage0, soft) | 60 | 0.5949 | 0.5949 | 0.0738 | 0.8990 | 0.3041 |
| UNet: SACA (after_merge0) | 60 | 0.6026 | 0.6026 | 0.1392 | 0.8972 | 0.2946 |
| UNet: SACA (after_stage1) | 60 | 0.5897 | 0.5897 | 0.1500 | 0.9044 | 0.3148 |
| UNet: Multi SACA (2 lớp) | 60 | 0.6030 | 0.6030 | 0.1340 | 0.8956 | 0.2926 |
| UNet: Multi SACA (4 lớp) | 60 | 0.5947 | 0.5947 | 0.1436 | 0.9046 | 0.3099 |
| Swin-UNet: Single View | 60 | 0.4043 | 0.4043 | 0.3694 | 0.9155 | 0.5112 |
| Swin-UNet: Dual View | 60 | 0.4123 | 0.4123 | 0.4156 | 0.9259 | 0.5136 |

---

## 3. Per-class Dice trên eval

Dice từng lớp: Necrotic-Core · Edema · Enhancing-Tumor · Background.

### 3.1 Tại epoch tốt nhất (region Dice TB)

| Cấu hình | Epoch | Necrotic-Core | Edema | Enhancing-Tumor | Background | PC-Macro |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 21 | 0.3894 | 0.7725 | 0.5361 | 0.9980 | 0.5660 |
| UNet: Dual View | 20 | 0.4114 | 0.8024 | 0.5705 | 0.9984 | 0.5947 |
| UNet: SACA (after_patch_embed) | 27 | 0.4099 | 0.8101 | 0.5706 | 0.9985 | 0.5969 |
| UNet: SACA (after_patch_embed, soft) | 27 | 0.4350 | 0.7995 | 0.5485 | 0.9983 | 0.5944 |
| UNet: SACA (after_stage0) | 22 | 0.4300 | 0.8039 | 0.5516 | 0.9983 | 0.5952 |
| UNet: SACA (after_stage0, soft) | 21 | 0.4401 | 0.7954 | 0.5669 | 0.9982 | 0.6008 |
| UNet: SACA (after_merge0) | 12 | 0.3765 | 0.8044 | 0.5657 | 0.9984 | 0.5822 |
| UNet: SACA (after_stage1) | 29 | 0.4506 | 0.7860 | 0.5575 | 0.9980 | 0.5981 |
| UNet: Multi SACA (2 lớp) | 31 | 0.4170 | 0.8054 | 0.5689 | 0.9984 | 0.5971 |
| UNet: Multi SACA (4 lớp) | 12 | 0.3810 | 0.8016 | 0.5680 | 0.9984 | 0.5835 |
| Swin-UNet: Single View | 16 | 0.1862 | 0.6551 | 0.3593 | 0.9974 | 0.4002 |
| Swin-UNet: Dual View | 13 | 0.1543 | 0.6822 | 0.3827 | 0.9973 | 0.4064 |

### 3.2 Tại epoch cuối cùng

| Cấu hình | Epoch | Necrotic-Core | Edema | Enhancing-Tumor | Background | PC-Macro |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 60 | 0.3904 | 0.7789 | 0.5190 | 0.9981 | 0.5628 |
| UNet: Dual View | 60 | 0.4239 | 0.7960 | 0.5492 | 0.9982 | 0.5897 |
| UNet: SACA (after_patch_embed) | 60 | 0.4594 | 0.7979 | 0.5552 | 0.9983 | 0.6041 |
| UNet: SACA (after_patch_embed, soft) | 60 | 0.4200 | 0.7980 | 0.5343 | 0.9982 | 0.5841 |
| UNet: SACA (after_stage0) | 60 | 0.4701 | 0.7991 | 0.5365 | 0.9982 | 0.6019 |
| UNet: SACA (after_stage0, soft) | 60 | 0.4313 | 0.8043 | 0.5490 | 0.9984 | 0.5949 |
| UNet: SACA (after_merge0) | 60 | 0.4548 | 0.8059 | 0.5470 | 0.9984 | 0.6026 |
| UNet: SACA (after_stage1) | 60 | 0.4332 | 0.7934 | 0.5423 | 0.9981 | 0.5897 |
| UNet: Multi SACA (2 lớp) | 60 | 0.4547 | 0.8000 | 0.5543 | 0.9983 | 0.6030 |
| UNet: Multi SACA (4 lớp) | 60 | 0.4476 | 0.7863 | 0.5501 | 0.9980 | 0.5947 |
| Swin-UNet: Single View | 60 | 0.1941 | 0.6890 | 0.3298 | 0.9975 | 0.4043 |
| Swin-UNet: Dual View | 60 | 0.1924 | 0.7052 | 0.3393 | 0.9976 | 0.4123 |

---

## 4. Suy giảm: Best epoch → Epoch 60

Δ càng nhỏ càng ổn định (ít overfit / ít mất hiệu năng cuối training).

| Cấu hình | Best Ep (Reg) | Dice TB best | Dice TB last | Δ Dice TB | Best Ep (PC) | PC-Macro best | PC-Macro last | Δ PC-Macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 21 | 0.724 | 0.716 | -0.0080 | 35 | 0.5706 | 0.5628 | -0.0079 |
| UNet: Dual View | 20 | 0.750 | 0.741 | -0.0096 | 15 | 0.6068 | 0.5897 | -0.0171 |
| UNet: SACA (after_patch_embed) | 27 | 0.755 | 0.744 | -0.0109 | 12 | 0.6187 | 0.6041 | -0.0146 |
| UNet: SACA (after_patch_embed, soft) | 27 | 0.739 | 0.725 | -0.0134 | 19 | 0.5961 | 0.5841 | -0.0120 |
| UNet: SACA (after_stage0) | 22 | 0.740 | 0.730 | -0.0095 | 14 | 0.6093 | 0.6019 | -0.0074 |
| UNet: SACA (after_stage0, soft) | 21 | 0.749 | 0.741 | -0.0082 | 42 | 0.6028 | 0.5949 | -0.0080 |
| UNet: SACA (after_merge0) | 12 | 0.751 | 0.740 | -0.0109 | 16 | 0.6082 | 0.6026 | -0.0057 |
| UNet: SACA (after_stage1) | 29 | 0.741 | 0.732 | -0.0087 | 29 | 0.5981 | 0.5897 | -0.0084 |
| UNet: Multi SACA (2 lớp) | 31 | 0.752 | 0.743 | -0.0092 | 30 | 0.6090 | 0.6030 | -0.0060 |
| UNet: Multi SACA (4 lớp) | 12 | 0.752 | 0.734 | -0.0174 | 17 | 0.6101 | 0.5947 | -0.0154 |
| Swin-UNet: Single View | 16 | 0.600 | 0.574 | -0.0261 | 11 | 0.4154 | 0.4043 | -0.0111 |
| Swin-UNet: Dual View | 13 | 0.611 | 0.578 | -0.0330 | 12 | 0.4301 | 0.4123 | -0.0178 |

---

## Đánh giá tổng hợp

- **Tốt nhất theo region (Dice TB)**: **UNet: SACA (after_patch_embed)** (epoch 27) — Dice TB **0.755**, WT 0.923 / TC 0.771 / ET 0.571.
- **Tốt nhất theo PC-Macro Dice**: **UNet: SACA (after_patch_embed)** (epoch 12) — PC-Macro **0.6187**.
- **Tốt nhất UNet + SACA**: **UNet: SACA (after_patch_embed)** (epoch 27) — Dice TB **0.755**.
- **UNet vs Swin-UNet** (trung bình Dice TB best): UNet **0.745** vs Swin **0.605** (chênh ~0.140).

---

## Nguồn dữ liệu

Mỗi thư mục `turmor_seg_flair/<exp>/`:

- `region_dice_log.csv` — WT/TC/ET Dice, IoU, HD95 (train & eval)
- `epoch_log.csv` — loss, macro Dice, PC-Macro Dice
- `per_class_dice_eval.csv` — Dice từng class trên eval
- `reports/epoch_reports.jsonl` — log JSON theo epoch
