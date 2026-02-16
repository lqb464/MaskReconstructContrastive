from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


_EMPTY_HANDLING_VALUES = {"exclude", "one"}


@dataclass
class DiceBuffers:
    """Running buffers for epoch-level Dice computation."""

    intersection: torch.Tensor
    denominator: torch.Tensor


def _validate_empty_handling(empty_handling: str) -> None:
    if empty_handling not in _EMPTY_HANDLING_VALUES:
        raise ValueError(f"empty_handling must be one of {_EMPTY_HANDLING_VALUES}, got {empty_handling}")


def _ensure_label_map(pred_or_logits: torch.Tensor) -> torch.Tensor:
    if pred_or_logits.ndim == 4:
        return torch.argmax(pred_or_logits, dim=1)
    if pred_or_logits.ndim >= 1:
        return pred_or_logits
    raise ValueError(
        f"pred_or_logits must be logits [B,C,H,W] or label tensor, got shape={tuple(pred_or_logits.shape)}"
    )


def init_dice_buffers(*, num_classes: int, device: torch.device) -> DiceBuffers:
    c = int(num_classes)
    if c < 2:
        raise ValueError(f"num_classes must be >= 2, got {c}")
    return DiceBuffers(
        intersection=torch.zeros((c,), dtype=torch.float64, device=device),
        denominator=torch.zeros((c,), dtype=torch.float64, device=device),
    )


def accumulate_intersection_union(
    pred_or_logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    *,
    buffers: Optional[DiceBuffers] = None,
) -> DiceBuffers:
    """
    Update running intersection/denominator buffers for multi-class Dice.

    Dice_c = 2*I_c / (|pred_c| + |tgt_c|)
    """
    pred = _ensure_label_map(pred_or_logits)
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} must match target shape {tuple(target.shape)}")

    c = int(num_classes)
    if buffers is None:
        buffers = init_dice_buffers(num_classes=c, device=pred.device)

    # Flatten once and accumulate class-wise counts.
    pred_f = pred.reshape(-1)
    tgt_f = target.reshape(-1)
    for cls in range(c):
        pred_c = pred_f == cls
        tgt_c = tgt_f == cls
        inter = (pred_c & tgt_c).sum(dtype=torch.float64)
        denom = pred_c.sum(dtype=torch.float64) + tgt_c.sum(dtype=torch.float64)
        buffers.intersection[cls] += inter
        buffers.denominator[cls] += denom

    return buffers


def finalize_dice(
    buffers: DiceBuffers,
    *,
    eps: float = 1e-6,
    empty_handling: str = "exclude",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert buffers to per-class Dice and validity mask.

    Returns:
      - dice: [C] float64
      - valid_mask: [C] bool (classes included in macro by default policy)
    """
    _validate_empty_handling(empty_handling)

    denom = buffers.denominator
    inter = buffers.intersection

    dice = torch.zeros_like(denom, dtype=torch.float64)
    valid = denom > 0

    if valid.any():
        dice[valid] = (2.0 * inter[valid]) / (denom[valid] + float(eps))

    if empty_handling == "one":
        dice[~valid] = 1.0
        valid_mask = torch.ones_like(valid, dtype=torch.bool)
    else:  # exclude
        valid_mask = valid

    return dice, valid_mask


def dice_per_class_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    *,
    include_bg: bool = False,
    empty_handling: str = "exclude",
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single-call convenience wrapper returning per-class Dice and valid mask.
    """
    buffers = accumulate_intersection_union(logits, target, num_classes)
    dice, valid_mask = finalize_dice(buffers, eps=eps, empty_handling=empty_handling)
    if not include_bg and dice.numel() > 0:
        valid_mask = valid_mask.clone()
        valid_mask[0] = False
    return dice, valid_mask


def macro_dice(
    dice_tensor: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    include_bg: bool = False,
) -> torch.Tensor:
    """
    Macro Dice over valid classes.
    Returns NaN if no class is valid for reduction.
    """
    if dice_tensor.ndim != 1 or valid_mask.ndim != 1:
        raise ValueError("dice_tensor and valid_mask must be rank-1")
    if dice_tensor.numel() != valid_mask.numel():
        raise ValueError("dice_tensor and valid_mask must have identical length")

    mask = valid_mask.clone()
    if not include_bg and mask.numel() > 0:
        mask[0] = False

    vals = dice_tensor[mask]
    if vals.numel() == 0:
        return torch.full((), float("nan"), device=dice_tensor.device, dtype=dice_tensor.dtype)
    return vals.mean()


def dice_summary(
    dice_tensor: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    include_bg: bool = False,
) -> dict[str, torch.Tensor | int]:
    """
    Return min/mean/max and number of valid classes for logging.
    """
    mask = valid_mask.clone()
    if not include_bg and mask.numel() > 0:
        mask[0] = False

    vals = dice_tensor[mask]
    if vals.numel() == 0:
        nan = torch.full((), float("nan"), device=dice_tensor.device, dtype=dice_tensor.dtype)
        return {
            "dice_min": nan,
            "dice_mean": nan,
            "dice_max": nan,
            "num_valid_classes": 0,
        }

    return {
        "dice_min": vals.min(),
        "dice_mean": vals.mean(),
        "dice_max": vals.max(),
        "num_valid_classes": int(mask.sum().item()),
    }


def format_class_dice_line(
    per_class_dice: torch.Tensor | dict[int, float | torch.Tensor],
    *,
    class_ids: Iterable[int] | None = None,
) -> str:
    """Compact formatter for per-class Dice preview lines."""
    chunks = []

    if isinstance(per_class_dice, dict):
        ids = sorted(per_class_dice.keys()) if class_ids is None else list(class_ids)
        for cid in ids:
            v = per_class_dice[int(cid)]
            fv = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            chunks.append(f"c{int(cid)}={fv:.4f}")
        return " ".join(chunks)

    if per_class_dice.ndim != 1:
        raise ValueError("per_class_dice tensor must be rank-1")

    ids = list(range(per_class_dice.numel())) if class_ids is None else list(class_ids)
    for cid in ids:
        fv = float(per_class_dice[int(cid)].item())
        chunks.append(f"c{int(cid)}={fv:.4f}")
    return " ".join(chunks)


__all__ = [
    "DiceBuffers",
    "init_dice_buffers",
    "accumulate_intersection_union",
    "finalize_dice",
    "dice_per_class_from_logits",
    "macro_dice",
    "dice_summary",
    "format_class_dice_line",
]
