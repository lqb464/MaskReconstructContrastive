from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sample_masks_random_patches(
    batch_size: int,
    *,
    image_size: int,
    patch_size: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Random patch masking (MAE-style): each batch item masks floor(mask_ratio * num_patches) patches.

    Returns pixel mask [B, 1, H, W] with 1 = masked.
    """
    h = w = int(image_size)
    p = int(patch_size)
    gh, gw = h // p, w // p
    num_patches = gh * gw
    num_masked = int(math.floor(float(mask_ratio) * num_patches))
    mask = torch.zeros((batch_size, 1, h, w), dtype=torch.float32, device=device)
    if batch_size <= 0 or num_masked <= 0:
        return mask

    for b in range(batch_size):
        perm = torch.randperm(num_patches, device=device)[:num_masked]
        patch_grid = torch.zeros(gh * gw, dtype=torch.float32, device=device)
        patch_grid[perm] = 1.0
        patch_grid = patch_grid.view(gh, gw)
        pixel = patch_grid.repeat_interleave(p, dim=0).repeat_interleave(p, dim=1)
        if pixel.shape[0] < h or pixel.shape[1] < w:
            pixel = F.pad(pixel, (0, max(w - pixel.shape[1], 0), 0, max(h - pixel.shape[0], 0)))
        mask[b, 0] = pixel[:h, :w]

    return mask


__all__ = ["sample_masks_random_patches"]
