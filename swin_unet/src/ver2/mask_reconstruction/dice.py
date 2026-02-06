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


__all__ = ["dice_coefficient", "soft_dice_loss"]
