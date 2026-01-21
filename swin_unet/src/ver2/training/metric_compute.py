from __future__ import annotations

from typing import Tuple, Optional

import torch

from swin_unet.src.ver2.common.losses import ssim_index


def update_recon_metrics(
    *,
    meter,
    x: torch.Tensor,
    x_flip: Optional[torch.Tensor],
    recon_raw_orig: torch.Tensor,
    recon_raw_flip: Optional[torch.Tensor],
    pixel_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Update MetricsAccumulator exactly as legacy trainer.

    Returns (recon_img_orig_metric, recon_img_flip_metric, diff_total, ssim_sum).
    """
    recon_img_orig_metric = torch.sigmoid(recon_raw_orig.clamp(-10, 10))

    if recon_raw_flip is None or x_flip is None:
        if recon_raw_flip is not None or x_flip is not None:
            raise ValueError("recon_raw_flip and x_flip must both be None or both be provided.")
        diff_orig = (x - recon_img_orig_metric).abs()
        diff_total = diff_orig.detach()
        ssim_orig = ssim_index(x.float(), recon_img_orig_metric.float())
        ssim_sum = float(ssim_orig.sum().item())
        meter.update(diff_total, pixel_mask, ssim_sum=ssim_sum)
        return recon_img_orig_metric, None, diff_total, ssim_sum

    recon_img_flip_metric = torch.sigmoid(recon_raw_flip.clamp(-10, 10))

    diff_orig = (x - recon_img_orig_metric).abs()
    diff_flip = (x_flip - recon_img_flip_metric).abs()
    diff_total = (0.5 * (diff_orig + diff_flip)).detach()

    ssim_orig = ssim_index(x.float(), recon_img_orig_metric.float())
    ssim_flip = ssim_index(x_flip.float(), recon_img_flip_metric.float())
    ssim_sum = float((0.5 * (ssim_orig + ssim_flip)).sum().item())

    meter.update(diff_total, pixel_mask, ssim_sum=ssim_sum)
    return recon_img_orig_metric, recon_img_flip_metric, diff_total, ssim_sum # no need to return 
