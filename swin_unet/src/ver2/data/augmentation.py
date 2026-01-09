# =============================================
# File: augmentation.py
# Masking strategies and data augmentation
# =============================================
from __future__ import annotations

import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import MaskConfig


def sample_masks_anti_mirror(batch_size: int, spec: 'MaskConfig', device: torch.device) -> torch.Tensor:
    """
    Sample random patch masks that avoid perfect left-right mirror symmetry
    
    Args:
        batch_size: Number of masks to generate
        spec: Mask configuration (patch_size, mask_ratio_side, image_size)
        device: Target device
        
    Returns:
        Binary mask tensor of shape (B, 1, H, W) where 1 indicates masked regions
    """
    H = W = spec.image_size
    P = spec.patch_size
    gh, _gw = spec.grid_size()
    hw = spec.half_grid_w()
    per_side = int(math.floor(spec.mask_ratio_side * gh * hw))
    
    mask = torch.zeros((batch_size, 1, H, W), dtype=torch.float32, device=device)
    
    for b in range(batch_size):
        # Sample patches from left half
        all_left = [(r, c) for r in range(gh) for c in range(hw)]
        left_sel = set(random.sample(all_left, per_side))
        
        # Exclude mirror positions from right half
        mirror_exclude = set((r, hw - 1 - c) for (r, c) in left_sel)
        all_right = [(r, c) for r in range(gh) for c in range(hw)]
        right_candidates = [rc for rc in all_right if rc not in mirror_exclude]
        
        # Sample from right half
        right_sel = set(random.sample(
            all_right if per_side > len(right_candidates) else right_candidates, 
            per_side
        ))
        
        # Fill mask for left patches
        for (r, c) in left_sel:
            hs = r * P
            ws = c * P
            mask[b, 0, hs:hs + P, ws:ws + P] = 1.0
        
        # Fill mask for right patches
        for (r, c) in right_sel:
            hs = r * P
            ws = (hw + c) * P
            mask[b, 0, hs:hs + P, ws:ws + P] = 1.0
    
    return mask


class HalfAug(nn.Module):
    """
    Data augmentation for hemisphere-based contrastive learning
    Applies random noise, jitter, and blur to brain images
    """
    def __init__(
        self, 
        p_noise: float = 0.7, 
        p_jitter: float = 0.7, 
        p_blur: float = 0.2,
        noise_std: float = 0.02, 
        jitter_strength: float = 0.1, 
        blur_kernel: int = 3
    ):
        super().__init__()
        self.p_noise = p_noise
        self.p_jitter = p_jitter
        self.p_blur = p_blur
        self.noise_std = noise_std
        self.jitter_strength = jitter_strength
        self.blur_kernel = blur_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to input tensor"""
        # Gaussian noise
        if self.p_noise > 0:
            mask = torch.rand(x.size(0), device=x.device) < self.p_noise
            if mask.any():
                noise = torch.randn_like(x[mask]) * self.noise_std
                x[mask] = torch.clamp(x[mask] + noise, 0.0, 1.0)
        
        # Brightness and contrast jitter
        if self.p_jitter > 0:
            mask = torch.rand(x.size(0), device=x.device) < self.p_jitter
            if mask.any():
                b_shift = (torch.rand(x[mask].size(0), 1, 1, 1, device=x.device) - 0.5) * 2 * self.jitter_strength
                c_scale = 1.0 + (torch.rand(x[mask].size(0), 1, 1, 1, device=x.device) - 0.5) * 2 * self.jitter_strength
                x[mask] = torch.clamp((x[mask] - 0.5) * c_scale + 0.5 + b_shift, 0.0, 1.0)
        
        # Blur
        if self.p_blur > 0:
            mask = torch.rand(x.size(0), device=x.device) < self.p_blur
            if mask.any():
                x_blur = F.avg_pool2d(x[mask], kernel_size=self.blur_kernel, stride=1, padding=self.blur_kernel // 2)
                x[mask] = x_blur
        
        return x


__all__ = [
    "sample_masks_anti_mirror",
    "HalfAug",
]