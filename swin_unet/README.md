## SwinUNet Dual View SSL with Plane Conditioning

### 0) Inputs and metadata

```text
Given:
  x              : [B, 1, H, W]                     (single slice)
  M              : [B, 1, H, W]                     (binary mask, patch level expanded)
  plane_one_hot  : [B, 2]
    axial   -> [0, 1]
    coronal -> [1, 0]

Construct 2 views with the SAME mask M:
  x1_masked = x * (1 - M)                           (view1, masked)
  x2_masked = flip_lr(x) * (1 - M)                  (view2, flipped then masked, mask NOT flipped)

Note:
  flip_lr means left right flip on width dimension
```

---

### 1) Plane conditioning module

```text
PlaneEmbed:
  p = MLP_plane(plane_one_hot) -> [B, C2]           (C2 matches Stage2 channel)

Injection rule (choose 1):
  A) Additive:   f = f + broadcast(p)
  B) FiLM:       f = f * gamma(p) + beta(p)

broadcast(p) means expand p to [B, H', W', C2] and add to each token feature
```

---

### 2) Model backbone overview

```text
Encoder is split into:
  EarlyEncView1: PatchEmbed + Stage0 + MergeDown0 + Stage1
  EarlyEncView2: PatchEmbed + Stage0 + MergeDown0 + Stage1

Shared trunk begins at Stage2:
  SharedEnc: MergeDown1 + Stage2 + MergeDown2 + Stage3

Decoder exists only for view1:
  Decoder: Up stages + Final upsample + Recon head

Contrastive head uses features AFTER shared trunk:
  z1 = pool(b1) ; z2 = pool(b2) ; proj -> InfoNCE
```

---

### 3) Detailed architecture diagram with tensor shapes

Assume:

1. patch_size = P (typical P=4)
2. Swin stages: Stage0, Stage1, Stage2, Stage3
3. channel progression: C0, C1=2C0, C2=2C1, C3=2C2
4. spatial progression (in feature map space):
   Stage0: H/P, W/P
   Stage1: H/(2P), W/(2P)
   Stage2: H/(4P), W/(4P)
   Stage3: H/(8P), W/(8P)

```text
============================== VIEW 1 BRANCH ==============================
x1_masked : [B, 1, H, W]
   |
   v
PatchEmbed_1 (P x P, linear proj)
f0_1 : [B, H/P, W/P, C0]
   |
   v
Stage0_1 (BasicLayer: Swin blocks x d0)
s0_1 : [B, H/P, W/P, C0]                 save skip0_1 = s0_1
   |
   v
PatchMerging_0_1 (down x2)
f1_1 : [B, H/(2P), W/(2P), C1]
   |
   v
Stage1_1 (BasicLayer: Swin blocks x d1)
s1_1 : [B, H/(2P), W/(2P), C1]           save skip1_1 = s1_1


============================== VIEW 2 BRANCH ==============================
x2_masked : [B, 1, H, W]
   |
   v
PatchEmbed_2 (P x P, linear proj)
f0_2 : [B, H/P, W/P, C0]
   |
   v
Stage0_2 (BasicLayer: Swin blocks x d0)
s0_2 : [B, H/P, W/P, C0]
   |
   v
PatchMerging_0_2 (down x2)
f1_2 : [B, H/(2P), W/(2P), C1]
   |
   v
Stage1_2 (BasicLayer: Swin blocks x d1)
s1_2 : [B, H/(2P), W/(2P), C1]


============================== MERGE POINT ===============================
We do NOT merge tensors into one. We merge weights:
  From Stage2 onward, BOTH views pass through the SAME shared modules.

Shared trunk input:
  shared_in_1 = s1_1
  shared_in_2 = s1_2

============================== SHARED TRUNK ===============================
For view1:
  PatchMerging_1 (shared) -> u2_1 : [B, H/(4P), W/(4P), C2]
  PlaneCondition at Stage2 entry:
    u2_1 = Cond(u2_1, p)            (p from plane_one_hot)
  Stage2 (shared) -> s2_1 : [B, H/(4P), W/(4P), C2]       save skip2_1 = s2_1

  PatchMerging_2 (shared) -> u3_1 : [B, H/(8P), W/(8P), C3]
  Stage3 (shared) -> b1 : [B, H/(8P), W/(8P), C3]         bottleneck view1

For view2:
  PatchMerging_1 (shared) -> u2_2 : [B, H/(4P), W/(4P), C2]
  PlaneCondition at Stage2 entry:
    u2_2 = Cond(u2_2, p)
  Stage2 (shared) -> s2_2 : [B, H/(4P), W/(4P), C2]

  PatchMerging_2 (shared) -> u3_2 : [B, H/(8P), W/(8P), C3]
  Stage3 (shared) -> b2 : [B, H/(8P), W/(8P), C3]         bottleneck view2


============================== CONTRASTIVE HEAD ===============================
GlobalPool over spatial:
  z1 = GAP(b1) : [B, C3]
  z2 = GAP(b2) : [B, C3]

Projection head (shared):
  c1 = MLP_proj(z1) : [B, D]
  c2 = MLP_proj(z2) : [B, D]

Contrastive loss:
  L_contrast = InfoNCE(c1, c2)     (positives are paired by sample index)


============================== DECODER (VIEW1 ONLY) ===============================
Decoder takes only view1 features and view1 skips:
  Inputs:
    bottleneck b1
    skip2_1 = s2_1
    skip1_1 = s1_1
    skip0_1 = s0_1

UpStage2:
  PatchExpand (up x2) -> d2_up : [B, H/(4P), W/(4P), C2]
  concat with skip2_1 -> proj -> Swin up blocks -> d2 : [B, H/(4P), W/(4P), C2]

UpStage1:
  PatchExpand (up x2) -> d1_up : [B, H/(2P), W/(2P), C1]
  concat with skip1_1 -> proj -> Swin up blocks -> d1 : [B, H/(2P), W/(2P), C1]

UpStage0:
  PatchExpand (up x2) -> d0_up : [B, H/P, W/P, C0]
  concat with skip0_1 -> proj -> Swin up blocks -> d0 : [B, H/P, W/P, C0]

Final upsample to full resolution:
  FinalPatchExpand (up xP) -> feat_full : [B, H, W, C_final]

Reconstruction head:
  x_hat = HeadRecon(feat_full) : [B, 1, H, W]

Reconstruction loss computed only on view1:
  L_recon = ReconLoss(x_hat, target_x1)
```

---

### 4) Loss and training objective

```text
Total loss:
  L = lambda_contrast * L_contrast + lambda_recon * L_recon

Where:
  L_contrast uses representations AFTER shared trunk (b1, b2 pooled)
  L_recon uses decoder output for view1 only
```

---

### 5) Minimal config block

```text
Dual view:
  view1 = masked
  view2 = flip_lr then apply same mask M

Split depth:
  split_to_stage = 1               (separate weights for PatchEmbed, Stage0, Stage1)

Shared trunk:
  shared_from_stage = 2            (Stage2, Stage3 shared)

Plane conditioning:
  plane_one_hot: axial [0,1], coronal [1,0]
  inject_stage = 2
  inject_method = FiLM or Add

SSL:
  contrastive_tap = after_shared_bottleneck
  pool = global average pool
  proj_head = 2 layer MLP

Decoder:
  decode_view = 1 only
  decoder_skips = from view1 path
```
