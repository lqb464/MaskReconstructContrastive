# =============================================
# File: masking.py
# Asymmetry (anti-mirror) patch masking
# Patch-level mask, expanded to pixel mask
# =============================================

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch


# -------------------------------------------------
# Mask specification
# -------------------------------------------------

@dataclass
class MaskSpec:
    patch_size: int = 16
    image_size: int = 256
    mask_ratio_side: float = 0.35

    def grid_size(self) -> Tuple[int, int]:
        gh = self.image_size // self.patch_size
        gw = self.image_size // self.patch_size
        return gh, gw

    def half_grid_w(self) -> int:
        return (self.image_size // 2) // self.patch_size

    def num_patches_side(self) -> int:
        gh, _ = self.grid_size()
        hw = self.half_grid_w()
        return gh * hw


# -------------------------------------------------
# Anti-mirror mask sampling
# -------------------------------------------------

def sample_masks_anti_mirror(
    batch_size: int,
    spec: MaskSpec,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns:
        pixel_mask: [B, 1, H, W]
        - mask == 1.0: masked
        - mask == 0.0: visible

    Logic:
        - Sample patches on left half
        - Sample patches on right half but forbid mirror positions
        - Expand patch mask -> pixel mask
    """
    H = W = spec.image_size
    P = spec.patch_size
    gh, _ = spec.grid_size()
    hw = spec.half_grid_w()

    per_side = int(math.floor(spec.mask_ratio_side * gh * hw))

    mask = torch.zeros(
        (batch_size, 1, H, W),
        dtype=torch.float32,
        device=device,
    )

    for b in range(batch_size):
        # ---- left half ----
        all_left = [(r, c) for r in range(gh) for c in range(hw)]
        left_sel = set(random.sample(all_left, per_side))

        # forbid mirror of left patches on right side
        mirror_exclude = set((r, hw - 1 - c) for (r, c) in left_sel)

        # ---- right half ----
        all_right = [(r, c) for r in range(gh) for c in range(hw)]
        right_candidates = [
            rc for rc in all_right if rc not in mirror_exclude
        ]
        right_sel = set(
            random.sample(
                all_right if per_side > len(right_candidates) else right_candidates,
                per_side,
            )
        )

        # ---- write pixel mask ----
        for (r, c) in left_sel:
            hs = r * P
            ws = c * P
            mask[b, 0, hs:hs + P, ws:ws + P] = 1.0

        for (r, c) in right_sel:
            hs = r * P
            ws = (hw + c) * P
            mask[b, 0, hs:hs + P, ws:ws + P] = 1.0

    return mask
