from __future__ import annotations

from typing import Callable, Tuple

import torch

from ..models.model_utils import flip_lr


def dual_view_loss_and_logits(
    recon1: torch.Tensor,
    recon2: torch.Tensor,
    target: torch.Tensor,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    flip_target: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if flip_target:
        target_v2 = torch.flip(target, dims=[-1])
    else:
        target_v2 = target
    loss_v1 = criterion(recon1, target)
    loss_v2 = criterion(recon2, target_v2)
    loss = 0.5 * (loss_v1 + loss_v2)
    recon2_aligned = flip_lr(recon2)
    logits = 0.5 * (recon1 + recon2_aligned)
    return loss, logits


__all__ = ["dual_view_loss_and_logits"]
