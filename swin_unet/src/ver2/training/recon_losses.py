from __future__ import annotations

import torch
import torch.nn.functional as F


def _foreground_weighted_bce_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    fg_eps: float = 0.02,
    fg_weight: float = 10.0,
) -> torch.Tensor:
    """Weighted BCEWithLogits where pixels with target > fg_eps get larger weight.

    Target is expected in [0, 1]. Returns unreduced loss map.
    """
    with torch.no_grad():
        w = torch.ones_like(target)
        w = torch.where(target > fg_eps, torch.full_like(w, fg_weight), w)
    return F.binary_cross_entropy_with_logits(logits, target, weight=w, reduction="none")


def masked_bce_logits_weighted(
    logits: torch.Tensor,
    target: torch.Tensor,
    pixel_mask: torch.Tensor,
    fg_eps: float = 0.02,
    fg_weight: float = 10.0,
) -> torch.Tensor:
    """BCE logits computed only on masked region (pixel_mask==1)."""
    loss_map = _foreground_weighted_bce_logits(logits, target, fg_eps=fg_eps, fg_weight=fg_weight)
    m = pixel_mask
    denom = m.sum().clamp(min=1.0)
    return (loss_map * m).sum() / denom


def mixed_bce_logits_weighted(
    logits: torch.Tensor,
    target: torch.Tensor,
    pixel_mask: torch.Tensor,
    fg_eps: float = 0.02,
    fg_weight: float = 10.0,
    alpha_mask: float = 1.0,
    beta_unmask: float = 0.2,
) -> torch.Tensor:
    """Weighted BCE logits computed on both masked and unmasked, with different weights."""
    loss_map = _foreground_weighted_bce_logits(logits, target, fg_eps=fg_eps, fg_weight=fg_weight)
    m = pixel_mask
    um = 1.0 - m
    masked = (loss_map * m).sum() / m.sum().clamp(min=1.0)
    unmasked = (loss_map * um).sum() / um.sum().clamp(min=1.0)
    return alpha_mask * masked + beta_unmask * unmasked
