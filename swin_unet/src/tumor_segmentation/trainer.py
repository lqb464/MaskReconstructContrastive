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

import numpy as np
import torch

from ..tissue_segmentation.trainer import TissueSegmentationTrainer
from .region_dice import (
    BRATS_REGION_CLASSES,
    RegionDiceBuffers,
    accumulate_region_dice,
    finalize_region_dice,
    finalize_region_iou,
    compute_hd95_2d,
    init_region_dice_buffers,
    region_dice_summary_line,
)


class RegionDiceCsvLogger:
    """Appends per-epoch region metrics (dice, iou, hd95) to a CSV file."""

    HEADERS = [
        "epoch", "split",
        "wt_dice", "tc_dice", "et_dice",
        "wt_iou", "tc_iou", "et_iou",
        "wt_hd95", "tc_hd95", "et_hd95"
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADERS)

    def append(self, epoch: int, split: str, region_dice: Dict[str, float], region_iou: Dict[str, float], region_hd95: Dict[str, float]) -> None:
        row = [
            int(epoch),
            str(split),
            region_dice.get("wt", float("nan")),
            region_dice.get("tc", float("nan")),
            region_dice.get("et", float("nan")),
            region_iou.get("wt", float("nan")),
            region_iou.get("tc", float("nan")),
            region_iou.get("et", float("nan")),
            region_hd95.get("wt", float("nan")),
            region_hd95.get("tc", float("nan")),
            region_hd95.get("et", float("nan")),
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
        self._region_hd95_lists: Dict[str, list[float]] = {}
        # Store last computed region dice for use in _post_epoch_hook
        self._last_train_region_dice: Dict[str, float] = {}
        self._last_eval_region_dice: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Hook overrides
    # ------------------------------------------------------------------

    def _on_epoch_start(self, training: bool) -> None:
        if training and hasattr(self.model, "set_encoder_trainable"):
            freeze_n = int(getattr(self.cfg.training, "freeze_encoder_epochs", 0))
            if freeze_n > 0:
                epoch = int(getattr(self.model, "current_epoch", 1))
                self.model.set_encoder_trainable(trainable=not (epoch <= freeze_n))
        if self.enable_region_dice:
            self._region_buffers = init_region_dice_buffers(device=self.device)
            self._region_hd95_lists = {
                "wt": [],
                "tc": [],
                "et": []
            }

    def _on_batch_end_with_preds(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        if not self.enable_region_dice or self._region_buffers is None:
            return
        if valid.any():
            # Standard region dice / iou accumulation via flattened tensors
            accumulate_region_dice(pred[valid], target[valid], self._region_buffers)

            # HD95 calculation slice-by-slice
            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy()

            for i in range(pred.size(0)):
                p_slice = pred_np[i]
                t_slice = target_np[i]
                v_slice = valid_np[i]

                # Mask out ignored/invalid pixels
                p_slice[~v_slice] = 0
                t_slice[~v_slice] = 0

                # Compute region masks (1=NCR, 2=ED, 3=ET after mode-3 label mapping)
                wt_pred = np.isin(p_slice, [1, 2, 3])
                wt_tgt  = np.isin(t_slice, [1, 2, 3])
                tc_pred = np.isin(p_slice, [1, 3])
                tc_tgt  = np.isin(t_slice, [1, 3])
                et_pred = (p_slice == 3)
                et_tgt  = (t_slice == 3)

                hd_wt = compute_hd95_2d(wt_pred, wt_tgt)
                hd_tc = compute_hd95_2d(tc_pred, tc_tgt)
                hd_et = compute_hd95_2d(et_pred, et_tgt)

                if not math.isnan(hd_wt):
                    self._region_hd95_lists["wt"].append(hd_wt)
                if not math.isnan(hd_tc):
                    self._region_hd95_lists["tc"].append(hd_tc)
                if not math.isnan(hd_et):
                    self._region_hd95_lists["et"].append(hd_et)

    def _on_epoch_end_stats(self, stats: dict) -> dict:
        if self.enable_region_dice and self._region_buffers is not None:
            stats["region_dice"] = finalize_region_dice(self._region_buffers)
            stats["region_iou"]  = finalize_region_iou(self._region_buffers)

            region_hd95 = {}
            for name, hd_list in self._region_hd95_lists.items():
                if len(hd_list) > 0:
                    region_hd95[name] = float(np.mean(hd_list))
                else:
                    region_hd95[name] = float("nan")
            stats["region_hd95"] = region_hd95
        else:
            stats["region_dice"] = {}
            stats["region_iou"] = {}
            stats["region_hd95"] = {}
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
        train_iou    = train_stats.get("region_iou", {})
        eval_iou     = eval_stats.get("region_iou", {})
        train_hd95   = train_stats.get("region_hd95", {})
        eval_hd95    = eval_stats.get("region_hd95", {})

        self._last_train_region_dice = train_region
        self._last_eval_region_dice  = eval_region

        if train_region:
            print(f"[train/region] Dice: {region_dice_summary_line(train_region)} | IoU: {region_dice_summary_line(train_iou)} | HD95: {region_dice_summary_line(train_hd95)}")

        if should_eval and eval_region:
            print(f"[eval/region]  Dice: {region_dice_summary_line(eval_region)} | IoU: {region_dice_summary_line(eval_iou)} | HD95: {region_dice_summary_line(eval_hd95)}")

        if self._region_log is not None:
            self._region_log.append(epoch, "train", train_region, train_iou, train_hd95)
            if should_eval:
                self._region_log.append(epoch, "eval", eval_region, eval_iou, eval_hd95)


__all__ = ["TumorSegmentationTrainer"]
