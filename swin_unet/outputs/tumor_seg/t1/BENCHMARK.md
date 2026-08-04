# Bảng so sánh kết quả phân đoạn u não (T1 only)

Nguồn: `MaskReconstructContrastive` → `turmor_seg_t1/` (16 cấu hình × 60 epoch, modality **T1**).
Số **in đậm** = tốt nhất trong nhóm UNet (1–8) hoặc Swin-UNet (9–16).

---

## 1. Bảng chính — Region Dice (WT / TC / ET)

Metric BraTS chuẩn. **Epoch tốt nhất** chọn theo Dice TB = (WT+TC+ET)/3 trên eval.

### 1.1 Epoch tốt nhất

| Cấu hình | Epoch | WT Dice | WT IoU | WT HD95 | TC Dice | TC IoU | TC HD95 | ET Dice | ET IoU | ET HD95 | Dice TB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 24 | 0.794 | 0.659 | 17.20 | 0.684 | 0.520 | 8.72 | 0.474 | 0.311 | 8.60 | 0.651 |
| UNet: Dual View | 27 | 0.820 | 0.695 | 13.42 | 0.715 | 0.557 | 7.65 | 0.508 | 0.340 | 7.74 | 0.681 |
| UNet: SACA (after_patch_embed) | 29 | 0.816 | 0.689 | 13.99 | 0.720 | 0.563 | 7.61 | 0.513 | 0.345 | 7.58 | 0.683 |
| UNet: SACA (after_stage0) | 31 | 0.820 | 0.695 | 13.56 | 0.711 | 0.552 | 7.21 | 0.510 | 0.342 | 7.11 | 0.681 |
| UNet: SACA (after_merge0) | 26 | 0.821 | 0.696 | 12.87 | **0.721** | **0.564** | 7.31 | 0.510 | 0.342 | 7.37 | 0.684 |
| UNet: SACA (after_stage1) | 28 | 0.821 | 0.696 | **12.73** | 0.718 | 0.560 | **7.13** | **0.515** | **0.347** | **7.09** | **0.685** |
| UNet: Multi SACA (2 lớp) | 32 | 0.820 | 0.695 | 13.50 | 0.721 | 0.563 | 7.24 | 0.512 | 0.344 | 7.23 | 0.684 |
| UNet: Multi SACA (4 lớp) | 32 | **0.821** | **0.697** | 13.88 | 0.718 | 0.561 | 7.50 | 0.513 | 0.345 | 7.54 | 0.684 |
| Swin-UNet: Single View | 18 | 0.725 | 0.568 | 20.84 | 0.530 | 0.361 | 14.18 | 0.335 | 0.201 | 13.53 | 0.530 |
| Swin-UNet: Dual View (Không SACA) | 20 | 0.740 | 0.587 | 18.24 | 0.552 | 0.382 | 10.77 | 0.356 | 0.217 | 10.34 | 0.550 |
| Swin-UNet: SACA (after_patch_embed) | 25 | 0.745 | 0.593 | 17.68 | 0.537 | 0.367 | 11.09 | 0.344 | 0.208 | 10.53 | 0.542 |
| Swin-UNet: SACA (after_stage0) | 19 | 0.742 | 0.590 | 18.01 | 0.543 | 0.373 | 10.92 | 0.355 | 0.216 | 10.50 | 0.547 |
| Swin-UNet: SACA (after_merge0) | 24 | 0.745 | 0.593 | 17.23 | **0.561** | **0.390** | 10.72 | **0.366** | **0.224** | 10.39 | **0.557** |
| Swin-UNet: SACA (after_stage1) | 26 | 0.745 | 0.593 | 17.50 | 0.543 | 0.373 | **10.57** | 0.351 | 0.213 | **10.17** | 0.546 |
| Swin-UNet: Multi SACA (2 lớp) | 18 | 0.742 | 0.590 | 17.97 | 0.557 | 0.386 | 11.34 | 0.361 | 0.221 | 10.94 | 0.553 |
| Swin-UNet: Multi SACA (4 lớp) | 21 | **0.749** | **0.599** | **17.15** | 0.546 | 0.376 | 10.64 | 0.356 | 0.217 | 10.34 | 0.551 |

### 1.2 Epoch cuối cùng (epoch 60)

| Cấu hình | Epoch | WT Dice | WT IoU | WT HD95 | TC Dice | TC IoU | TC HD95 | ET Dice | ET IoU | ET HD95 | Dice TB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 60 | 0.792 | 0.656 | 18.25 | 0.669 | 0.503 | 9.23 | 0.452 | 0.292 | 8.81 | 0.638 |
| UNet: Dual View | 60 | 0.817 | 0.690 | 13.37 | 0.705 | 0.545 | 7.16 | 0.489 | 0.324 | 6.93 | 0.670 |
| UNet: SACA (after_patch_embed) | 60 | 0.818 | 0.692 | 13.37 | 0.704 | 0.543 | 7.02 | 0.494 | 0.328 | 6.78 | 0.672 |
| UNet: SACA (after_stage0) | 60 | 0.817 | 0.690 | **13.05** | 0.703 | 0.542 | 7.03 | 0.497 | 0.331 | 6.79 | 0.672 |
| UNet: SACA (after_merge0) | 60 | 0.816 | 0.689 | 13.29 | 0.708 | 0.548 | 7.06 | 0.492 | 0.326 | 6.92 | 0.672 |
| UNet: SACA (after_stage1) | 60 | 0.817 | 0.690 | 13.19 | 0.700 | 0.539 | 6.95 | 0.490 | 0.325 | **6.73** | 0.669 |
| UNet: Multi SACA (2 lớp) | 60 | 0.817 | 0.691 | 13.14 | **0.710** | **0.551** | **6.89** | 0.498 | 0.332 | 6.75 | **0.675** |
| UNet: Multi SACA (4 lớp) | 60 | **0.820** | **0.695** | 13.21 | 0.705 | 0.544 | 7.31 | **0.500** | **0.333** | 7.14 | 0.675 |
| Swin-UNet: Single View | 60 | 0.712 | 0.553 | 21.98 | 0.494 | 0.328 | 12.67 | 0.296 | 0.174 | 11.89 | 0.501 |
| Swin-UNet: Dual View (Không SACA) | 60 | 0.724 | 0.568 | 17.56 | 0.504 | 0.337 | 10.35 | 0.303 | 0.179 | 9.77 | 0.510 |
| Swin-UNet: SACA (after_patch_embed) | 60 | 0.727 | 0.571 | 17.34 | 0.500 | 0.333 | 10.36 | 0.297 | 0.175 | 9.75 | 0.508 |
| Swin-UNet: SACA (after_stage0) | 60 | 0.722 | 0.564 | 17.65 | 0.497 | 0.331 | 10.38 | 0.293 | 0.171 | 9.76 | 0.504 |
| Swin-UNet: SACA (after_merge0) | 60 | 0.726 | 0.570 | 17.45 | 0.509 | 0.341 | **10.04** | 0.307 | 0.181 | 9.53 | 0.514 |
| Swin-UNet: SACA (after_stage1) | 60 | 0.726 | 0.570 | 17.32 | 0.502 | 0.335 | 10.25 | 0.301 | 0.177 | 9.68 | 0.510 |
| Swin-UNet: Multi SACA (2 lớp) | 60 | **0.732** | **0.577** | **16.84** | **0.514** | **0.346** | 10.09 | **0.309** | **0.183** | **9.39** | **0.518** |
| Swin-UNet: Multi SACA (4 lớp) | 60 | 0.731 | 0.576 | 16.93 | 0.505 | 0.338 | 10.13 | 0.301 | 0.177 | 9.70 | 0.513 |

---

## 2. PC-Macro Dice & Loss (metric chọn checkpoint)

PC-Macro Dice = macro Dice trên các class có mặt, **loại Background**.

### 2.1 Epoch tốt nhất (theo PC-Macro)

| Cấu hình | Epoch tốt nhất | PC-Macro Dice | Macro Dice (all) | Eval Loss | Train PC-Macro | Gap (Train−Eval) |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 35 | 0.5086 | 0.5086 | 0.1855 | 0.8157 | 0.3071 |
| UNet: Dual View | 14 | 0.5454 | 0.5454 | 0.1184 | 0.6804 | 0.1350 |
| UNet: SACA (after_patch_embed) | 26 | 0.5423 | 0.5423 | 0.1495 | 0.7805 | 0.2382 |
| UNet: SACA (after_stage0) | 16 | 0.5466 | 0.5466 | 0.1261 | 0.7047 | 0.1581 |
| UNet: SACA (after_merge0) | 40 | 0.5422 | 0.5422 | 0.1673 | 0.8195 | 0.2773 |
| UNet: SACA (after_stage1) | 36 | 0.5393 | 0.5393 | 0.1750 | 0.8217 | 0.2824 |
| UNet: Multi SACA (2 lớp) | 42 | 0.5394 | 0.5394 | 0.1694 | 0.8201 | 0.2807 |
| UNet: Multi SACA (4 lớp) | 19 | 0.5396 | 0.5396 | 0.1362 | 0.7455 | 0.2059 |
| Swin-UNet: Single View | 29 | 0.3869 | 0.3869 | 0.2776 | 0.7795 | 0.3926 |
| Swin-UNet: Dual View (Không SACA) | 32 | 0.4046 | 0.4046 | 0.3046 | 0.8214 | 0.4169 |
| Swin-UNet: SACA (after_patch_embed) | 30 | 0.4059 | 0.4059 | 0.2919 | 0.8141 | 0.4082 |
| Swin-UNet: SACA (after_stage0) | 32 | 0.4022 | 0.4022 | 0.3121 | 0.8273 | 0.4252 |
| Swin-UNet: SACA (after_merge0) | 16 | 0.4089 | 0.4089 | 0.1921 | 0.6947 | 0.2858 |
| Swin-UNet: SACA (after_stage1) | 28 | 0.4041 | 0.4041 | 0.2766 | 0.8011 | 0.3969 |
| Swin-UNet: Multi SACA (2 lớp) | 23 | 0.4100 | 0.4100 | 0.2344 | 0.7630 | 0.3529 |
| Swin-UNet: Multi SACA (4 lớp) | 31 | 0.4049 | 0.4049 | 0.2867 | 0.8160 | 0.4111 |

### 2.2 Epoch cuối cùng

| Cấu hình | Epoch | PC-Macro Dice | Macro Dice (all) | Eval Loss | Train PC-Macro | Gap (Train−Eval) |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 60 | 0.5005 | 0.5005 | 0.2403 | 0.8595 | 0.3590 |
| UNet: Dual View | 60 | 0.5320 | 0.5320 | 0.1982 | 0.8376 | 0.3056 |
| UNet: SACA (after_patch_embed) | 60 | 0.5333 | 0.5333 | 0.2038 | 0.8453 | 0.3120 |
| UNet: SACA (after_stage0) | 60 | 0.5330 | 0.5330 | 0.1983 | 0.8396 | 0.3066 |
| UNet: SACA (after_merge0) | 60 | 0.5354 | 0.5354 | 0.2000 | 0.8423 | 0.3068 |
| UNet: SACA (after_stage1) | 60 | 0.5295 | 0.5295 | 0.2161 | 0.8552 | 0.3256 |
| UNet: Multi SACA (2 lớp) | 60 | 0.5302 | 0.5302 | 0.1980 | 0.8388 | 0.3086 |
| UNet: Multi SACA (4 lớp) | 60 | 0.5298 | 0.5298 | 0.1988 | 0.8546 | 0.3248 |
| Swin-UNet: Single View | 60 | 0.3681 | 0.3681 | 0.4971 | 0.8757 | 0.5076 |
| Swin-UNet: Dual View (Không SACA) | 60 | 0.3865 | 0.3865 | 0.5726 | 0.8949 | 0.5084 |
| Swin-UNet: SACA (after_patch_embed) | 60 | 0.3896 | 0.3896 | 0.5794 | 0.8987 | 0.5091 |
| Swin-UNet: SACA (after_stage0) | 60 | 0.3853 | 0.3853 | 0.6133 | 0.9015 | 0.5163 |
| Swin-UNet: SACA (after_merge0) | 60 | 0.3862 | 0.3862 | 0.5548 | 0.8985 | 0.5124 |
| Swin-UNet: SACA (after_stage1) | 60 | 0.3872 | 0.3872 | 0.5839 | 0.8974 | 0.5103 |
| Swin-UNet: Multi SACA (2 lớp) | 60 | 0.3899 | 0.3899 | 0.5332 | 0.8924 | 0.5025 |
| Swin-UNet: Multi SACA (4 lớp) | 60 | 0.3903 | 0.3903 | 0.5500 | 0.8953 | 0.5051 |

---

## 3. Per-class Dice trên eval

Dice từng lớp: Necrotic-Core · Edema · Enhancing-Tumor · Background.

### 3.1 Tại epoch tốt nhất (region Dice TB)

| Cấu hình | Epoch | Necrotic-Core | Edema | Enhancing-Tumor | Background | PC-Macro |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 24 | 0.4327 | 0.5979 | 0.4740 | 0.9955 | 0.5015 |
| UNet: Dual View | 27 | 0.4308 | 0.6263 | 0.5075 | 0.9962 | 0.5215 |
| UNet: SACA (after_patch_embed) | 29 | 0.4799 | 0.6294 | 0.5119 | 0.9959 | 0.5404 |
| UNet: SACA (after_stage0) | 31 | 0.4408 | 0.6362 | 0.5100 | 0.9961 | 0.5290 |
| UNet: SACA (after_merge0) | 26 | 0.4651 | 0.6321 | 0.5097 | 0.9962 | 0.5356 |
| UNet: SACA (after_stage1) | 28 | 0.4421 | 0.6355 | 0.5151 | 0.9962 | 0.5309 |
| UNet: Multi SACA (2 lớp) | 32 | 0.4192 | 0.6382 | 0.5117 | 0.9961 | 0.5230 |
| UNet: Multi SACA (4 lớp) | 32 | 0.4497 | 0.6327 | 0.5110 | 0.9961 | 0.5311 |
| Swin-UNet: Single View | 18 | 0.3406 | 0.4769 | 0.3352 | 0.9940 | 0.3842 |
| Swin-UNet: Dual View (Không SACA) | 20 | 0.3431 | 0.5002 | 0.3562 | 0.9944 | 0.3998 |
| Swin-UNet: SACA (after_patch_embed) | 25 | 0.3510 | 0.5134 | 0.3444 | 0.9946 | 0.4029 |
| Swin-UNet: SACA (after_stage0) | 19 | 0.2973 | 0.5087 | 0.3553 | 0.9945 | 0.3871 |
| Swin-UNet: SACA (after_merge0) | 24 | 0.3427 | 0.5145 | 0.3662 | 0.9948 | 0.4078 |
| Swin-UNet: SACA (after_stage1) | 26 | 0.3331 | 0.5124 | 0.3508 | 0.9947 | 0.3988 |
| Swin-UNet: Multi SACA (2 lớp) | 18 | 0.3400 | 0.4997 | 0.3614 | 0.9946 | 0.4004 |
| Swin-UNet: Multi SACA (4 lớp) | 21 | 0.2909 | 0.5140 | 0.3562 | 0.9947 | 0.3870 |

### 3.2 Tại epoch cuối cùng

| Cấu hình | Epoch | Necrotic-Core | Edema | Enhancing-Tumor | Background | PC-Macro |
| --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 60 | 0.4531 | 0.5965 | 0.4519 | 0.9956 | 0.5005 |
| UNet: Dual View | 60 | 0.4717 | 0.6350 | 0.4893 | 0.9961 | 0.5320 |
| UNet: SACA (after_patch_embed) | 60 | 0.4669 | 0.6394 | 0.4936 | 0.9961 | 0.5333 |
| UNet: SACA (after_stage0) | 60 | 0.4684 | 0.6336 | 0.4970 | 0.9962 | 0.5330 |
| UNet: SACA (after_merge0) | 60 | 0.4782 | 0.6360 | 0.4921 | 0.9961 | 0.5354 |
| UNet: SACA (after_stage1) | 60 | 0.4618 | 0.6365 | 0.4904 | 0.9961 | 0.5295 |
| UNet: Multi SACA (2 lớp) | 60 | 0.4532 | 0.6381 | 0.4993 | 0.9961 | 0.5302 |
| UNet: Multi SACA (4 lớp) | 60 | 0.4539 | 0.6366 | 0.4988 | 0.9962 | 0.5298 |
| Swin-UNet: Single View | 60 | 0.3186 | 0.4895 | 0.2961 | 0.9941 | 0.3681 |
| Swin-UNet: Dual View (Không SACA) | 60 | 0.3490 | 0.5076 | 0.3030 | 0.9947 | 0.3865 |
| Swin-UNet: SACA (after_patch_embed) | 60 | 0.3590 | 0.5126 | 0.2973 | 0.9947 | 0.3896 |
| Swin-UNet: SACA (after_stage0) | 60 | 0.3548 | 0.5084 | 0.2926 | 0.9947 | 0.3853 |
| Swin-UNet: SACA (after_merge0) | 60 | 0.3390 | 0.5126 | 0.3069 | 0.9947 | 0.3862 |
| Swin-UNet: SACA (after_stage1) | 60 | 0.3494 | 0.5108 | 0.3013 | 0.9947 | 0.3872 |
| Swin-UNet: Multi SACA (2 lớp) | 60 | 0.3447 | 0.5163 | 0.3088 | 0.9948 | 0.3899 |
| Swin-UNet: Multi SACA (4 lớp) | 60 | 0.3556 | 0.5138 | 0.3014 | 0.9948 | 0.3903 |

---

## 4. Suy giảm: Best epoch → Epoch 60

Δ càng nhỏ càng ổn định (ít overfit / ít mất hiệu năng cuối training).

| Cấu hình | Best Ep (Reg) | Dice TB best | Dice TB last | Δ Dice TB | Best Ep (PC) | PC-Macro best | PC-Macro last | Δ PC-Macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 24 | 0.651 | 0.638 | -0.0129 | 35 | 0.5086 | 0.5005 | -0.0081 |
| UNet: Dual View | 27 | 0.681 | 0.670 | -0.0106 | 14 | 0.5454 | 0.5320 | -0.0134 |
| UNet: SACA (after_patch_embed) | 29 | 0.683 | 0.672 | -0.0113 | 26 | 0.5423 | 0.5333 | -0.0090 |
| UNet: SACA (after_stage0) | 31 | 0.681 | 0.672 | -0.0083 | 16 | 0.5466 | 0.5330 | -0.0136 |
| UNet: SACA (after_merge0) | 26 | 0.684 | 0.672 | -0.0121 | 40 | 0.5422 | 0.5354 | -0.0067 |
| UNet: SACA (after_stage1) | 28 | 0.685 | 0.669 | -0.0156 | 36 | 0.5393 | 0.5295 | -0.0098 |
| UNet: Multi SACA (2 lớp) | 32 | 0.684 | 0.675 | -0.0089 | 42 | 0.5394 | 0.5302 | -0.0092 |
| UNet: Multi SACA (4 lớp) | 32 | 0.684 | 0.675 | -0.0093 | 19 | 0.5396 | 0.5298 | -0.0099 |
| Swin-UNet: Single View | 18 | 0.530 | 0.501 | -0.0292 | 29 | 0.3869 | 0.3681 | -0.0189 |
| Swin-UNet: Dual View (Không SACA) | 20 | 0.550 | 0.510 | -0.0391 | 32 | 0.4046 | 0.3865 | -0.0180 |
| Swin-UNet: SACA (after_patch_embed) | 25 | 0.542 | 0.508 | -0.0341 | 30 | 0.4059 | 0.3896 | -0.0163 |
| Swin-UNet: SACA (after_stage0) | 19 | 0.547 | 0.504 | -0.0430 | 32 | 0.4022 | 0.3853 | -0.0169 |
| Swin-UNet: SACA (after_merge0) | 24 | 0.557 | 0.514 | -0.0434 | 16 | 0.4089 | 0.3862 | -0.0227 |
| Swin-UNet: SACA (after_stage1) | 26 | 0.546 | 0.510 | -0.0363 | 28 | 0.4041 | 0.3872 | -0.0170 |
| Swin-UNet: Multi SACA (2 lớp) | 18 | 0.553 | 0.518 | -0.0354 | 23 | 0.4100 | 0.3899 | -0.0201 |
| Swin-UNet: Multi SACA (4 lớp) | 21 | 0.551 | 0.513 | -0.0380 | 31 | 0.4049 | 0.3903 | -0.0146 |

---

## 5. Generalization — Train vs Eval (tại epoch tốt nhất region)

Gap = Train − Eval. Gap lớn → model fit train mạnh hơn eval.

| Cấu hình | Epoch | Train Dice TB | Eval Dice TB | Gap Reg | Train PC-Macro | Eval PC-Macro | Gap PC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UNet: Single View | 24 | 0.831 | 0.651 | 0.1802 | 0.7674 | 0.5015 | 0.2658 |
| UNet: Dual View | 27 | 0.839 | 0.681 | 0.1580 | 0.7744 | 0.5215 | 0.2528 |
| UNet: SACA (after_patch_embed) | 29 | 0.851 | 0.683 | 0.1682 | 0.7929 | 0.5404 | 0.2525 |
| UNet: SACA (after_stage0) | 31 | 0.850 | 0.681 | 0.1694 | 0.7895 | 0.5290 | 0.2605 |
| UNet: SACA (after_merge0) | 26 | 0.838 | 0.684 | 0.1543 | 0.7742 | 0.5356 | 0.2386 |
| UNet: SACA (after_stage1) | 28 | 0.853 | 0.685 | 0.1687 | 0.7950 | 0.5309 | 0.2641 |
| UNet: Multi SACA (2 lớp) | 32 | 0.853 | 0.684 | 0.1685 | 0.7919 | 0.5230 | 0.2689 |
| UNet: Multi SACA (4 lớp) | 32 | 0.864 | 0.684 | 0.1797 | 0.8071 | 0.5311 | 0.2760 |
| Swin-UNet: Single View | 18 | 0.765 | 0.530 | 0.2347 | 0.6835 | 0.3842 | 0.2993 |
| Swin-UNet: Dual View (Không SACA) | 20 | 0.812 | 0.550 | 0.2629 | 0.7396 | 0.3998 | 0.3397 |
| Swin-UNet: SACA (after_patch_embed) | 25 | 0.845 | 0.542 | 0.3028 | 0.7833 | 0.4029 | 0.3804 |
| Swin-UNet: SACA (after_stage0) | 19 | 0.808 | 0.547 | 0.2616 | 0.7337 | 0.3871 | 0.3466 |
| Swin-UNet: SACA (after_merge0) | 24 | 0.841 | 0.557 | 0.2834 | 0.7773 | 0.4078 | 0.3695 |
| Swin-UNet: SACA (after_stage1) | 26 | 0.849 | 0.546 | 0.3030 | 0.7880 | 0.3988 | 0.3892 |
| Swin-UNet: Multi SACA (2 lớp) | 18 | 0.794 | 0.553 | 0.2404 | 0.7143 | 0.4004 | 0.3139 |
| Swin-UNet: Multi SACA (4 lớp) | 21 | 0.819 | 0.551 | 0.2685 | 0.7481 | 0.3870 | 0.3611 |

---

## Đánh giá tổng hợp

- **Tốt nhất theo region (Dice TB)**: **UNet: SACA (after_stage1)** (epoch 28) — Dice TB **0.685**, WT 0.821 / TC 0.718 / ET 0.515.
- **Tốt nhất theo PC-Macro Dice**: **UNet: SACA (after_stage0)** (epoch 16) — PC-Macro **0.5466**.
- **UNet vs Swin-UNet** (trung bình Dice TB best): UNet **0.679** vs Swin **0.547** (chênh ~0.132).

---

## Nguồn dữ liệu

Mỗi thư mục `turmor_seg_t1/<exp>/`:

- `region_dice_log.csv` — WT/TC/ET Dice, IoU, HD95 (train & eval)
- `epoch_log.csv` — loss, macro Dice, PC-Macro Dice
- `per_class_dice_eval.csv` — Dice từng class trên eval
- `reports/epoch_reports.jsonl` — log JSON theo epoch
