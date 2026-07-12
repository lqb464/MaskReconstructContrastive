from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

from ..data.augmentation import sample_masks_anti_mirror


def _run_with_mask_seed(fn: Callable[[], torch.Tensor], mask_seed: int, device: torch.device) -> torch.Tensor:
    state = torch.get_rng_state()
    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device)
    torch.manual_seed(int(mask_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed(int(mask_seed))
    out = fn()
    torch.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)
    return out


def _sample_hemisphere_mask(batch_size: int, cfg_mask, device: torch.device) -> torch.Tensor:
    """Hemisphere anti-mirror patch mask (same family as swin_unet ver4)."""
    return sample_masks_anti_mirror(batch_size, cfg_mask, device)


def prepare_inputs(
    batch: Dict,
    *,
    device: torch.device,
    cfg_mask,
    backbone: str,
    mask_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare x, plane_one_hot, pixel_mask for MAE (masked) or VAE (no mask)."""
    x = batch["input"].to(device, non_blocking=True)

    plane = batch.get("plane_one_hot", None)
    if plane is None:
        plane = torch.tensor([0.0, 1.0], device=device).view(1, 2).repeat(x.size(0), 1)
    else:
        plane = plane.to(device, non_blocking=True)

    bb = str(backbone).lower()
    b, _, h, w = x.shape

    if bb == "vae":
        pixel_mask = torch.zeros((b, 1, h, w), device=device, dtype=torch.float32)
        return x, plane, pixel_mask

    if mask_seed is not None:
        pixel_mask = _run_with_mask_seed(
            lambda: _sample_hemisphere_mask(b, cfg_mask, device),
            int(mask_seed),
            device,
        )
    elif getattr(cfg_mask, "enable_masking", True):
        pixel_mask = _sample_hemisphere_mask(b, cfg_mask, device)
    else:
        pixel_mask = torch.zeros((b, 1, h, w), device=device, dtype=torch.float32)

    return x, plane, pixel_mask


def get_val_batch(loader, batch_index: int) -> Dict:
    """Return a fixed validation batch by index (deterministic order, shuffle=False)."""
    if batch_index < 0:
        raise ValueError(f"vis_batch_index must be >= 0, got {batch_index}")
    for i, batch in enumerate(loader):
        if i == batch_index:
            return batch
    raise IndexError(f"val loader has fewer than {batch_index + 1} batches")


__all__ = ["prepare_inputs", "get_val_batch"]
