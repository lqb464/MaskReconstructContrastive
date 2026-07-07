"""
region_dice.py

BraTS 2021 standard region-level Dice metrics.

After label_mode=3 remapping (IDs {0,1,2,4} -> {0,1,2,3}), the encoded classes are:
    0 = Background
    1 = Necrotic Core  (original BraTS label 1)
    2 = Edema          (original BraTS label 2)
    3 = Enhancing Tumor (original BraTS label 4)

BraTS evaluation regions:
    WT (Whole Tumor)      = encoded classes {1, 2, 3}
    TC (Tumor Core)       = encoded classes {1, 3}
    ET (Enhancing Tumor)  = encoded class   {3}

Region Dice is computed as binary Dice on the merged region mask:
    Dice_region = 2 * |pred_region ∩ tgt_region| / (|pred_region| + |tgt_region|)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Set, Tuple

import torch

# Encoded class IDs for each BraTS region (after mode-3 label mapping)
BRATS_REGION_CLASSES: Dict[str, FrozenSet[int]] = {
    "wt": frozenset({1, 2, 3}),  # Whole Tumor
    "tc": frozenset({1, 3}),     # Tumor Core
    "et": frozenset({3}),        # Enhancing Tumor
}

REGION_DISPLAY_NAMES: Dict[str, str] = {
    "wt": "WT (Whole Tumor)",
    "tc": "TC (Tumor Core)",
    "et": "ET (Enhancing Tumor)",
}


@dataclass
class RegionDiceBuffers:
    """
    Per-region running intersection/denominator buffers for epoch-level Dice.

    Each entry maps region name -> (intersection, denominator) as float64 scalars.
    """

    intersection: Dict[str, torch.Tensor]
    denominator: Dict[str, torch.Tensor]


def init_region_dice_buffers(
    *,
    device: torch.device,
    regions: Optional[Dict[str, FrozenSet[int]]] = None,
) -> RegionDiceBuffers:
    """
    Initialise zeroed region dice buffers.

    Args:
        device: torch device to place buffers on.
        regions: mapping of region_name -> set of encoded class IDs.
                 Defaults to BRATS_REGION_CLASSES.
    """
    if regions is None:
        regions = BRATS_REGION_CLASSES
    intersection: Dict[str, torch.Tensor] = {}
    denominator: Dict[str, torch.Tensor] = {}
    for name in regions:
        intersection[name] = torch.zeros((), dtype=torch.float64, device=device)
        denominator[name] = torch.zeros((), dtype=torch.float64, device=device)
    return RegionDiceBuffers(intersection=intersection, denominator=denominator)


def accumulate_region_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    buffers: RegionDiceBuffers,
    *,
    regions: Optional[Dict[str, FrozenSet[int]]] = None,
) -> RegionDiceBuffers:
    """
    Accumulate binary region intersection/denominator from a batch of predictions.

    Args:
        pred:    integer label tensor, arbitrary shape (flattened internally).
        target:  integer label tensor, same shape as pred.
        buffers: RegionDiceBuffers to accumulate into.
        regions: mapping of region_name -> set of encoded class IDs.
                 Defaults to BRATS_REGION_CLASSES.

    Returns:
        Updated buffers (in-place modification).
    """
    if regions is None:
        regions = BRATS_REGION_CLASSES

    if pred.shape != target.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} must match target shape {tuple(target.shape)}"
        )

    pred_f = pred.reshape(-1)
    tgt_f  = target.reshape(-1)

    for name, class_ids in regions.items():
        pred_mask = torch.zeros_like(pred_f, dtype=torch.bool)
        tgt_mask  = torch.zeros_like(tgt_f,  dtype=torch.bool)
        for cid in class_ids:
            pred_mask = pred_mask | (pred_f == cid)
            tgt_mask  = tgt_mask  | (tgt_f  == cid)

        inter = (pred_mask & tgt_mask).sum(dtype=torch.float64)
        denom = pred_mask.sum(dtype=torch.float64) + tgt_mask.sum(dtype=torch.float64)

        buffers.intersection[name] = buffers.intersection[name] + inter
        buffers.denominator[name]  = buffers.denominator[name]  + denom

    return buffers


def finalize_region_dice(
    buffers: RegionDiceBuffers,
    *,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute final Dice scores for each region from accumulated buffers.

    Returns:
        dict mapping region name (e.g. 'wt', 'tc', 'et') -> Dice value in [0, 1].
        NaN is returned for a region with zero denominator (no voxels predicted or targeted).
    """
    results: Dict[str, float] = {}
    for name in buffers.intersection:
        inter = float(buffers.intersection[name].item())
        denom = float(buffers.denominator[name].item())
        if denom < eps:
            results[name] = float("nan")
        else:
            results[name] = (2.0 * inter) / (denom + eps)
    return results


def region_dice_summary_line(region_dice: Dict[str, float]) -> str:
    """Format region dice values as a compact single-line string for logging."""
    parts = []
    for name, display in REGION_DISPLAY_NAMES.items():
        val = region_dice.get(name, float("nan"))
        if val != val:  # NaN check
            parts.append(f"{name.upper()}=nan")
        else:
            parts.append(f"{name.upper()}={val:.4f}")
    return "  ".join(parts)


__all__ = [
    "BRATS_REGION_CLASSES",
    "REGION_DISPLAY_NAMES",
    "RegionDiceBuffers",
    "init_region_dice_buffers",
    "accumulate_region_dice",
    "finalize_region_dice",
    "region_dice_summary_line",
]
