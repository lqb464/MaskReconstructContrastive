
from __future__ import annotations

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
    gh, gw = spec.grid_size()
    hw = spec.half_grid_w()
    per_side = int(math.floor(spec.mask_ratio_side * gh * hw))
    mask = torch.zeros((batch_size, 1, H, W), dtype=torch.float32, device=device)
    if batch_size <= 0 or per_side <= 0:
        return mask

    total_left = gh * hw

    left_scores = torch.rand((batch_size, total_left), device=device)
    left_idx = left_scores.topk(per_side, dim=1).indices

    r_left = left_idx // hw
    c_left = left_idx % hw
    mirror_idx = r_left * hw + (hw - 1 - c_left)

    exclude = torch.zeros((batch_size, total_left), dtype=torch.bool, device=device)
    exclude.scatter_(1, mirror_idx, True)

    right_scores = torch.rand((batch_size, total_left), device=device)
    right_scores_excl = right_scores.masked_fill(exclude, float("-inf"))
    right_idx_excl = right_scores_excl.topk(per_side, dim=1).indices
    right_idx_all = right_scores.topk(per_side, dim=1).indices

    num_candidates = total_left - exclude.sum(dim=1)
    use_all = (per_side > num_candidates).view(batch_size, 1).expand(-1, per_side)
    right_idx = torch.where(use_all, right_idx_all, right_idx_excl)

    r_right = right_idx // hw
    c_right = right_idx % hw

    mask_grid = torch.zeros((batch_size, gh * gw), dtype=torch.float32, device=device)
    left_pos = r_left * gw + c_left
    right_pos = r_right * gw + (c_right + hw)
    mask_grid.scatter_(1, left_pos, 1.0)
    mask_grid.scatter_(1, right_pos, 1.0)
    mask_grid = mask_grid.view(batch_size, gh, gw)

    mask_patch = mask_grid.repeat_interleave(P, dim=1).repeat_interleave(P, dim=2)
    if mask_patch.shape[1] < H or mask_patch.shape[2] < W:
        pad_b = max(H - mask_patch.shape[1], 0)
        pad_r = max(W - mask_patch.shape[2], 0)
        mask_patch = F.pad(mask_patch, (0, pad_r, 0, pad_b))
    mask = mask_patch[:, :H, :W].unsqueeze(1).contiguous()
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

        if self.p_noise > 0:
            mask = torch.rand(x.size(0), device=x.device) < self.p_noise
            if mask.any():
                noise = torch.randn_like(x[mask]) * self.noise_std
                x[mask] = torch.clamp(x[mask] + noise, 0.0, 1.0)

        if self.p_jitter > 0:
            mask = torch.rand(x.size(0), device=x.device) < self.p_jitter
            if mask.any():
                b_shift = (torch.rand(x[mask].size(0), 1, 1, 1, device=x.device) - 0.5) * 2 * self.jitter_strength
                c_scale = 1.0 + (torch.rand(x[mask].size(0), 1, 1, 1, device=x.device) - 0.5) * 2 * self.jitter_strength
                x[mask] = torch.clamp((x[mask] - 0.5) * c_scale + 0.5 + b_shift, 0.0, 1.0)

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
