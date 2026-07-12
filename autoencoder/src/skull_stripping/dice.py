from __future__ import annotations

import torch

def dice_coefficient(pred_prob: torch.Tensor, target: torch.Tensor, threshold: float | None = None, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice coefficient between prediction probabilities and binary target.
    If threshold is provided, predictions are binarized for the metric.
    Returns mean dice over batch.
    """
    if threshold is not None:
        pred = (pred_prob >= threshold).float()
    else:
        pred = pred_prob

    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    dice = (2.0 * intersection + eps) / denom
    return dice.mean()

def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss on logits (1 - dice(sigmoid(logits), target)).
    """
    prob = torch.sigmoid(logits)
    target = target.float()

    intersection = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    dice = (2.0 * intersection + eps) / denom
    loss = 1.0 - dice
    return loss.mean()

def soft_dice_from_probs(pred_prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft dice computed on probabilities. Returns per-sample dice (B,).
    """
    pred = pred_prob.flatten(1)
    tgt = target.flatten(1)
    intersection = (pred * tgt).sum(dim=1)
    denom = pred.sum(dim=1) + tgt.sum(dim=1) + eps
    dice = (2.0 * intersection + eps) / denom
    return dice

def soft_dice_loss_total(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dice = soft_dice_from_probs(prob, target, eps=eps)
    return (1.0 - dice).mean()

def _safe_region_dice(prob: torch.Tensor, target: torch.Tensor, region_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute dice on a region; if region has zero target mass and prediction mass is also zero, treat dice as 1,
    else 0. Returns per-sample dice.
    """
    region_mask = region_mask.float()
    pred_r = prob * region_mask
    tgt_r = target * region_mask
    tgt_sum = tgt_r.flatten(1).sum(dim=1)
    pred_sum = pred_r.flatten(1).sum(dim=1)
    intersection = (pred_r * tgt_r).flatten(1).sum(dim=1)
    dice = (2.0 * intersection + eps) / (pred_sum + tgt_sum + eps)

    empty = tgt_sum <= 0
    dice = torch.where(empty, torch.where(pred_sum <= 0, torch.ones_like(dice), torch.zeros_like(dice)), dice)
    return dice

def soft_dice_loss_by_region(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """
    Returns tuple: (loss_total, loss_masked, loss_unmasked)
    masked region is target==1, unmasked is target==0.
    """
    prob = torch.sigmoid(logits)
    m = (target > 0.5).float()
    dice_total = soft_dice_from_probs(prob, target, eps=eps)
    dice_masked = _safe_region_dice(prob, target, m, eps=eps)
    dice_unmasked = _safe_region_dice(prob, target, 1.0 - m, eps=eps)
    return (1.0 - dice_total).mean(), (1.0 - dice_masked).mean(), (1.0 - dice_unmasked).mean()

__all__ = [
    "dice_coefficient",
    "soft_dice_loss",
    "soft_dice_from_probs",
    "soft_dice_loss_total",
    "soft_dice_loss_by_region",
]
