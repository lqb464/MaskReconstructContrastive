from __future__ import annotations

from typing import Tuple

import torch

from swin_unet.src.ver2.common.losses import ssim_index


def update_recon_metrics(
    *,
    meter,
    x: torch.Tensor,
    x_flip: torch.Tensor,
    recon_raw_orig: torch.Tensor,
    recon_raw_flip: torch.Tensor,
    pixel_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Update MetricsAccumulator exactly as legacy trainer.

    Returns (recon_img_orig_metric, recon_img_flip_metric, diff_total, ssim_sum).
    """
    recon_img_orig_metric = torch.sigmoid(recon_raw_orig.clamp(-10, 10))
    recon_img_flip_metric = torch.sigmoid(recon_raw_flip.clamp(-10, 10))

    diff_orig = (x - recon_img_orig_metric).abs()
    diff_flip = (x_flip - recon_img_flip_metric).abs()
    diff_total = (0.5 * (diff_orig + diff_flip)).detach()

    ssim_orig = ssim_index(x.float(), recon_img_orig_metric.float())
    ssim_flip = ssim_index(x_flip.float(), recon_img_flip_metric.float())
    ssim_sum = float((0.5 * (ssim_orig + ssim_flip)).sum().item())

    meter.update(diff_total, pixel_mask, ssim_sum=ssim_sum)
    return recon_img_orig_metric, recon_img_flip_metric, diff_total, ssim_sum # no need to return 
