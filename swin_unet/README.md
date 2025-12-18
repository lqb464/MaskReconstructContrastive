```
Inputs:
  x : [B,1,H,W]
  M : [B,1,H,W]                 (same mask for both views, mask NOT flipped)
  plane_one_hot : [B,2]         axial [0,1], coronal [1,0]

Views:
  x1 = x * (1 - M)
  x2 = flip_lr(x) * (1 - M)

==================== DUAL ENCODER (split_to_stage = 1) ====================

View1 early:
  PatchEmbed_1 -> Stage0_1 -> Merge0_1 -> Stage1_1 -> s1_1
  also keep skip0_1 = s0_1

View2 early:
  PatchEmbed_2 -> Stage0_2 -> Merge0_2 -> Stage1_2 -> s1_2
  also keep skip0_2 = s0_2

==================== SHARED TRUNK (shared_from_stage = 2) ==================

Shared trunk modules: Merge1 -> PlaneCond -> Stage2 -> Merge2 -> Stage3

Run twice with shared weights:
  (s2_1, b1) = SharedTrunk(s1_1, plane_one_hot)
  (s2_2, b2) = SharedTrunk(s1_2, plane_one_hot)

Keep skips:
  view1: skip2_1 = s2_1, skip1_1 = s1_1, skip0_1 = s0_1
  view2: skip2_2 = s2_2, skip1_2 = s1_2, skip0_2 = s0_2

==================== CONTRASTIVE HEAD (after merge) ========================

z1 = Proj( GAP(b1) )
z2 = Proj( GAP(b2) )
L_contrast = InfoNCE(z1, z2)

==================== DUAL DECODER (SwinUnet standard) ======================

Decoder is Swin style (PatchExpand + concat skip + proj + Swin blocks)

Decoder view1:
  d2_1 = Up2(b1, skip2_1)
  d1_1 = Up1(d2_1, skip1_1)
  d0_1 = Up0(d1_1, skip0_1)
  feat1 = FinalUp(d0_1)
  xhat1 = ReconHead(feat1)

Decoder view2:
  d2_2 = Up2(b2, skip2_2)
  d1_2 = Up1(d2_2, skip1_2)
  d0_2 = Up0(d1_2, skip0_2)
  feat2 = FinalUp(d0_2)
  xhat2 = ReconHead(feat2)

Recon losses:
  L_recon1 = loss(xhat1, target1)
  L_recon2 = loss(xhat2, target2)   (target2 = flip_lr(target1))
Total:
  L = λc*L_contrast + λr1*L_recon1 + λr2*L_recon2
```