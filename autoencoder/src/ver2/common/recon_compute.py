from __future__ import annotations

from typing import Tuple, Optional

import torch

from .losses import (
    masked_bce_logits_weighted,
    masked_l1_loss,
    mixed_bce_logits_weighted,
    mixed_l1_loss,
)

def compute_recon_losses(
    *,
    recon_raw_orig: torch.Tensor,
    recon_raw_flip: Optional[torch.Tensor],
    x: torch.Tensor,
    x_flip: Optional[torch.Tensor],
    pixel_mask: torch.Tensor,
    training_cfg,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reconstruction losses exactly as in legacy trainer.

    Returns: (loss_recon_orig, loss_recon_flip, loss_recon_total)
    """
    recon_loss_type = getattr(training_cfg, "recon_loss", "weighted_bce_logits")
    fg_eps = float(getattr(training_cfg, "fg_eps", 0.02))
    fg_weight = float(getattr(training_cfg, "fg_weight", 10.0))

    if recon_raw_flip is None or x_flip is None:
        if recon_raw_flip is not None or x_flip is not None:
            raise ValueError("recon_raw_flip and x_flip must both be None or both be provided.")
        if recon_loss_type == "weighted_bce_logits":
            if training_cfg.enable_masked_loss:
                loss_recon_orig = masked_bce_logits_weighted(
                    recon_raw_orig, x, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight
                )
            else:
                loss_recon_orig = mixed_bce_logits_weighted(
                    recon_raw_orig, x, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight
                )
        else:
            recon_img_orig = torch.sigmoid(recon_raw_orig.clamp(-10, 10))
            if training_cfg.enable_masked_loss:
                loss_recon_orig = masked_l1_loss(recon_img_orig, x, pixel_mask)
            else:
                loss_recon_orig = mixed_l1_loss(recon_img_orig, x, pixel_mask)

        loss_recon_flip = torch.zeros_like(loss_recon_orig)
        loss_recon_total = loss_recon_orig
        return loss_recon_orig, loss_recon_flip, loss_recon_total

    if recon_loss_type == "weighted_bce_logits":
        if training_cfg.enable_masked_loss:
            loss_recon_orig = masked_bce_logits_weighted(
                recon_raw_orig, x, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight
            )
            loss_recon_flip = masked_bce_logits_weighted(
                recon_raw_flip, x_flip, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight
            )
        else:
            loss_recon_orig = mixed_bce_logits_weighted(
                recon_raw_orig, x, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight
            )
            loss_recon_flip = mixed_bce_logits_weighted(
                recon_raw_flip, x_flip, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight
            )
    else:
        recon_img_orig = torch.sigmoid(recon_raw_orig.clamp(-10, 10))
        recon_img_flip = torch.sigmoid(recon_raw_flip.clamp(-10, 10))

        if training_cfg.enable_masked_loss:
            loss_recon_orig = masked_l1_loss(recon_img_orig, x, pixel_mask)
            loss_recon_flip = masked_l1_loss(recon_img_flip, x_flip, pixel_mask)
        else:
            loss_recon_orig = mixed_l1_loss(recon_img_orig, x, pixel_mask)
            loss_recon_flip = mixed_l1_loss(recon_img_flip, x_flip, pixel_mask)

    loss_recon_total = loss_recon_orig + loss_recon_flip
    return loss_recon_orig, loss_recon_flip, loss_recon_total
