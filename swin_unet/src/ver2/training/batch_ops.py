from __future__ import annotations

from typing import Dict, Tuple

import torch

# Keep legacy imports working (trainer/eval might run from different CWDs)
try:
    from augmentation import sample_masks_anti_mirror
except Exception:  # pragma: no cover
    from phase1.data.augmentation import sample_masks_anti_mirror


def prepare_inputs(
    batch: Dict,
    *,
    device: torch.device,
    cfg_mask,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare x, plane_one_hot, pixel_mask exactly as in legacy trainer.

    This is a move-only helper to remove duplication between train/val/visualize.
    """
    x = batch["input"].to(device, non_blocking=True)

    plane = batch.get("plane_one_hot", None)
    if plane is None:
        plane = torch.tensor([0.0, 1.0], device=device).view(1, 2).repeat(x.size(0), 1)
    else:
        plane = plane.to(device, non_blocking=True)

    pixel_mask = sample_masks_anti_mirror(x.size(0), cfg_mask, device)
    return x, plane, pixel_mask
