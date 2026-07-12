"""
trainer.py

TumorSegmentationTrainer — extends TissueSegmentationTrainer with BraTS
region-level Dice metrics (WT/TC/ET) computed during training and evaluation.

Region metrics are logged to:
  <out_dir>/region_dice_log.csv   (epoch, split, wt_dice, tc_dice, et_dice)
and printed to console alongside per-class Dice each epoch.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Optional

import torch

from ..tissue_segmentation.trainer import TissueSegmentationTrainer
from .region_dice import (
    BRATS_REGION_CLASSES,
    RegionDiceBuffers,
    accumulate_region_dice,
    finalize_region_dice,
    init_region_dice_buffers,
    region_dice_summary_line,
)


class RegionDiceCsvLogger:
    """Appends per-epoch region dice to a CSV file."""

    HEADERS = ["epoch", "split", "wt_dice", "tc_dice", "et_dice"]

    def __init__(self, path: Path) -> None:
        self.path = path
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADERS)

    def append(self, epoch: int, split: str, region_dice: Dict[str, float]) -> None:
        row = [
            int(epoch),
            str(split),
            region_dice.get("wt", float("nan")),
            region_dice.get("tc", float("nan")),
            region_dice.get("et", float("nan")),
        ]
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)


class TumorSegmentationTrainer(TissueSegmentationTrainer):
    """
    TissueSegmentationTrainer extended with BraTS region-level Dice.

    Additional constructor argument:
        enable_region_dice (bool): enable WT/TC/ET region dice (default True).

    Additional outputs:
        <out_dir>/region_dice_log.csv
    """

    def __init__(self, *, enable_region_dice: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.enable_region_dice = bool(enable_region_dice)

        if self.enable_region_dice:
            self._region_log = RegionDiceCsvLogger(self.out_dir / "region_dice_log.csv")
        else:
            self._region_log = None

        # Per-epoch region buffers — initialised in _on_epoch_start
        self._region_buffers: Optional[RegionDiceBuffers] = None
        # Store last computed region dice for use in _post_epoch_hook
        self._last_train_region_dice: Dict[str, float] = {}
        self._last_eval_region_dice: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Hook overrides
    # ------------------------------------------------------------------

    def _on_epoch_start(self, training: bool) -> None:
        if self.enable_region_dice:
            self._region_buffers = init_region_dice_buffers(device=self.device)

    def _on_batch_end_with_preds(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        if not self.enable_region_dice or self._region_buffers is None:
            return
        if valid.any():
            accumulate_region_dice(pred[valid], target[valid], self._region_buffers)

    def _on_epoch_end_stats(self, stats: dict) -> dict:
        if self.enable_region_dice and self._region_buffers is not None:
            stats["region_dice"] = finalize_region_dice(self._region_buffers)
        else:
            stats["region_dice"] = {}
        return stats

    def _post_epoch_hook(
        self,
        *,
        epoch: int,
        train_stats: dict,
        eval_stats: dict,
        should_eval: bool,
    ) -> None:
        if not self.enable_region_dice:
            return

        train_region = train_stats.get("region_dice", {})
        eval_region  = eval_stats.get("region_dice", {})

        self._last_train_region_dice = train_region
        self._last_eval_region_dice  = eval_region

        if train_region:
            print(f"[train/region] {region_dice_summary_line(train_region)}")

        if should_eval and eval_region:
            print(f"[eval/region]  {region_dice_summary_line(eval_region)}")

        if self._region_log is not None:
            self._region_log.append(epoch, "train", train_region)
            if should_eval:
                self._region_log.append(epoch, "eval", eval_region)


__all__ = ["TumorSegmentationTrainer"]
