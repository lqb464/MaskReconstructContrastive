# SwinUNet Dual View SSL

(Masked Image Modeling + Contrastive Learning)

## 1. Inputs

```
x              : [B, 1, H, W]
mask           : [B, 1, H, W]        same mask for both views (mask NOT flipped)
plane_one_hot  : [B, 2]              axial [0,1], coronal [1,0]
```

## 2. Dual Views Construction

```
x1 = x * (1 - mask)
x2 = flip_lr(x) * (1 - mask)
```

* Mask is identical for both views
* Only the image is flipped, not the mask

---

## 3. Dual Encoder (split_to_stage = 1)

Each view has its **own early encoder**. No weight sharing before `stage2`.

### View 1 Early Encoder

```
patch_embed_1
 → stage0_1
 → merge0_1
 → stage1_1 → s1_1

skip0_1 = output(stage0_1)
```

### View 2 Early Encoder

```
patch_embed_2
 → stage0_2
 → merge0_2
 → stage1_2 → s1_2

skip0_2 = output(stage0_2)
```

---

## 4. Shared Trunk (shared_from_stage = 2)

From `stage2` onward, **all modules are shared** between the two views.

### Shared modules sequence

```
merge1
 → plane_cond
 → stage2
 → merge2
 → stage3
```

### Applied independently per view

```
(s2_1, bottleneck_1) = SharedTrunk(s1_1, plane_one_hot)
(s2_2, bottleneck_2) = SharedTrunk(s1_2, plane_one_hot)
```

* `s2_*` is used for decoder skip
* `bottleneck_*` is the deepest feature

---

## 5. Contrastive Branch (model returns embeddings only)

The model **does not compute contrastive loss**.
It only returns `(z1, z2)` according to the selected position.

### Supported contrastive positions

```
contrastive_position ∈ {
  "stage1",      // after stage1_*
  "stage2",      // after stage2 (shared)
  "bottleneck"   // after stage3
}
```

### Embedding extraction

```
z1 = proj_cX( GAP(feature_1) )
z2 = proj_cX( GAP(feature_2) )
```

Projection head used:

* `proj_c1` for `stage1`
* `proj_c2` for `stage2`
* `proj_c3` for `bottleneck`

### Supported contrastive losses (trainer side)

```
contrast_loss.type ∈ {
  "infonce",
  "vicreg"
}
```

Example:

```
L_contrast = ContrastiveLoss(z1, z2)
```

---

## 6. Decoder (Reconstruction Path)

### Shared high-level decoder

```
d2_1 = up2_shared(bottleneck_1, s2_1)
d2_2 = up2_shared(bottleneck_2, s2_2)
```

### Split low-level decoders

#### View 1 Branch

```
d1_1   = up1_v1(d2_1, s1_1)
d0_1   = up0_v1(d1_1, skip0_1)
feat1  = final_up_v1(d0_1)
xhat1  = recon_head_v1(feat1)
```

#### View 2 Branch

```
d1_2   = up1_v2(d2_2, s1_2)
d0_2   = up0_v2(d1_2, skip0_2)
feat2  = final_up_v2(d0_2)
xhat2  = recon_head_v2(feat2)
```

---

## 7. Reconstruction Loss

```
target1 = x
target2 = flip_lr(x)

L_recon1 = recon_loss(xhat1, target1)
L_recon2 = recon_loss(xhat2, target2)
```

* Reconstruction loss is computed **per view**
* Targets are correctly aligned with flipping

---

## 8. Total Loss (trainer)

```
L = λ_contrast * L_contrast
  + λ_recon1   * L_recon1
  + λ_recon2   * L_recon2
```

* Model outputs raw predictions and embeddings
* Trainer is fully responsible for loss computation

---

## 9. SACA (Spatially Aware Cross Attention)

SACA can be enabled at **exactly one position** per run.

```
saca_position ∈ {
  "after_patch_emb",
  "after_merge0",
  "after_stage1"
}
```

Behavior:

* Applied symmetrically to both views
* Executed before entering the shared trunk
* Disabled entirely if `enable_saca = false`

