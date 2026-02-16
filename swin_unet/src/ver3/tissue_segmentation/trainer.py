from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config.experiment import ExperimentConfig
from ..data.augmentation import sample_masks_anti_mirror
from ..models.swin_unet_dualview_ssl import SwinUNetDualViewSSL, flip_lr
from ..training.ckpt_io import save_checkpoint
from ..training.utils import ensure_dir
from .dice import (
    accumulate_intersection_union,
    dice_summary,
    finalize_dice,
    format_class_dice_line,
    init_dice_buffers,
    macro_dice,
)
from .visualization import save_val_visualization_grid

log = logging.getLogger(__name__)


class EpochLogger:
    HEADERS = [
        "epoch",
        "train_loss",
        "train_macro_dice",
        "train_dice_min",
        "train_dice_mean",
        "train_dice_max",
        "train_num_valid_classes",
        "eval_loss",
        "eval_macro_dice",
        "eval_dice_min",
        "eval_dice_mean",
        "eval_dice_max",
        "eval_num_valid_classes",
        "lr",
    ]

    def __init__(self, path: Path):
        self.path = path
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADERS)

    def append(self, row: Dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row[h] for h in self.HEADERS])


def _parse_ce_class_weights(spec: str, num_classes: int) -> torch.Tensor | None:
    s = (spec or "").strip()
    if not s:
        return None
    parts = [x.strip() for x in s.split(",") if x.strip()]
    vals = [float(x) for x in parts]
    if len(vals) != int(num_classes):
        raise ValueError(
            f"--ce-class-weights expects {num_classes} values, got {len(vals)}: {parts}"
        )
    return torch.tensor(vals, dtype=torch.float32)


class TissueSegmentationTrainer:
    def __init__(
        self,
        *,
        model: SwinUNetDualViewSSL,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        out_dir: Path,
        cfg: ExperimentConfig,
        num_classes: int,
        class_names: Optional[Dict[int, str]] = None,
        dice_include_bg: bool = False,
        vis_every: int = 0,
        vis_num: int = 4,
        disable_tqdm: bool = False,
        val_every: int = 1,
        align_flip_target: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.num_classes = int(num_classes)
        self.class_names = class_names or {}
        self.dice_include_bg = bool(dice_include_bg)
        self.dice_empty_handling = "one" if bool(getattr(cfg.tissue, "dice_empty_as_one", False)) else "exclude"
        self.ignore_index = int(getattr(cfg.tissue, "ignore_index", -100))
        self.vis_every = int(vis_every)
        self.vis_num = max(0, min(4, int(vis_num)))
        self.disable_tqdm = bool(disable_tqdm)
        self.val_every = max(1, int(val_every))
        self.align_flip_target = bool(align_flip_target)
        self.vis_enabled = self.vis_every > 0 and self.vis_num > 0

        self.save_latest_every = max(1, int(getattr(cfg.logging, "save_latest_every", 1)))
        self.save_best_after_epoch = max(0, int(getattr(cfg.logging, "save_best_after_epoch", 0)))
        self.save_best_every = max(1, int(getattr(cfg.logging, "save_best_every", 1)))

        self._assert_model_contract()

        ce_w = _parse_ce_class_weights(str(getattr(cfg.tissue, "ce_class_weights", "")), self.num_classes)
        if ce_w is not None:
            ce_w = ce_w.to(device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=ce_w, ignore_index=self.ignore_index)
        print(
            f"[loss] CrossEntropyLoss(ignore_index={self.ignore_index}, "
            f"class_weights={'on' if ce_w is not None else 'off'})"
        )

        self.use_amp = bool(cfg.training.amp) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        self._warmup_epochs = int(getattr(cfg.training, "warmup_epochs", 0))
        self._min_lr = float(getattr(cfg.training, "min_lr", 0.0))
        self._base_lr = float(cfg.training.lr)
        self._scheduler_total_epochs = int(cfg.training.epochs)
        self.lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self._build_scheduler(self._scheduler_total_epochs)

        self.out_dir = ensure_dir(Path(out_dir))
        self.ckpt_dir = ensure_dir(self.out_dir / "checkpoints")
        self.vis_dir = ensure_dir(self.out_dir / "vis") if self.vis_enabled else None
        self.logger = EpochLogger(self.out_dir / "epoch_log.csv")

        self.best_macro_dice = float("-inf")
        self._val_vis_batch: Optional[dict[str, torch.Tensor]] = None

    def _assert_model_contract(self) -> None:
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >=2, got {self.num_classes}")

        for attr in ("recon_head_v1", "recon_head_v2"):
            head = getattr(self.model, attr, None)
            if head is None:
                raise RuntimeError(f"Model missing {attr}; tissue task requires reconstruction head for logits.")
            if not hasattr(head, "__len__") or len(head) < 1 or not isinstance(head[-1], torch.nn.Conv2d):
                raise RuntimeError(f"Unexpected {attr} structure; expected nn.Sequential ending with Conv2d")
            out_ch = int(head[-1].out_channels)
            if out_ch != int(self.num_classes):
                raise RuntimeError(
                    f"{attr} output channels ({out_ch}) != num_classes ({self.num_classes})"
                )

    def _build_scheduler(self, total_epochs: int) -> None:
        total_epochs = max(1, int(total_epochs))
        warmup = self._warmup_epochs
        min_lr = self._min_lr
        base_lr = self._base_lr

        def lr_lambda(epoch: int):
            if warmup > 0 and epoch < warmup:
                return float(epoch + 1) / float(warmup)
            t = epoch - warmup
            T = max(1, total_epochs - warmup)
            cosine = 0.5 * (1 + math.cos(math.pi * t / T))
            return (min_lr / base_lr) + (1 - min_lr / base_lr) * cosine

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self._scheduler_total_epochs = total_epochs

    def _sample_pixel_mask(self, x: torch.Tensor) -> torch.Tensor | None:
        if not bool(getattr(self.cfg.mask, "enable_masking", False)):
            return None
        return sample_masks_anti_mirror(x.size(0), self.cfg.mask, x.device)

    def _validate_targets(self, target: torch.Tensor) -> None:
        if target.dtype != torch.long:
            raise TypeError(f"target dtype must be torch.long for CrossEntropyLoss, got {target.dtype}")

        mask = target != self.ignore_index
        if mask.any():
            vals = target[mask]
            tmin = int(vals.min().item())
            tmax = int(vals.max().item())
            if tmin < 0 or tmax >= self.num_classes:
                raise ValueError(
                    f"Target labels out of range [0,{self.num_classes - 1}] (excluding ignore_index={self.ignore_index}): "
                    f"min={tmin}, max={tmax}"
                )

    def _forward_views(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        plane_one_hot: torch.Tensor,
        pixel_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        recon1, recon2, _, _ = self.model(x, pixel_mask, plane_one_hot)
        if recon1 is None:
            raise RuntimeError("Model returned recon1=None in tissue segmentation training.")
        if recon1.ndim != 4 or int(recon1.shape[1]) != self.num_classes:
            raise RuntimeError(
                f"Unexpected recon1 shape {tuple(recon1.shape)}, expected [B,{self.num_classes},H,W]."
            )

        loss1 = self.criterion(recon1, target)
        views: list[tuple[torch.Tensor, torch.Tensor]] = [(recon1, target)]

        if recon2 is not None:
            target2 = flip_lr(target.unsqueeze(1)).squeeze(1) if self.align_flip_target else target
            if recon2.ndim != 4 or int(recon2.shape[1]) != self.num_classes:
                raise RuntimeError(
                    f"Unexpected recon2 shape {tuple(recon2.shape)}, expected [B,{self.num_classes},H,W]."
                )
            loss2 = self.criterion(recon2, target2)
            loss = 0.5 * (loss1 + loss2)
            views.append((recon2, target2))
        else:
            loss = loss1

        return loss, views

    def _run_epoch(
        self,
        loader: DataLoader,
        *,
        training: bool,
        capture_vis: bool = False,
    ) -> dict[str, float | torch.Tensor | int | None]:
        if training:
            self.model.train()
            desc = "Train"
        else:
            self.model.eval()
            desc = "Eval"

        loss_sum = 0.0
        steps = 0
        dice_buf = init_dice_buffers(num_classes=self.num_classes, device=self.device)
        vis_payload: dict[str, torch.Tensor] | None = None

        progress = loader if self.disable_tqdm else tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for batch_idx, batch in enumerate(progress):
            x = batch["input"].to(self.device, non_blocking=True)
            y = batch["target"].to(self.device, non_blocking=True)
            plane = batch["plane_one_hot"].to(self.device, non_blocking=True)
            pixel_mask = self._sample_pixel_mask(x)

            self._validate_targets(y)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, views = self._forward_views(x, y, plane, pixel_mask)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected in {'train' if training else 'eval'} at step {batch_idx}: {loss}")

            if training:
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.cfg.training.grad_clip))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.cfg.training.grad_clip))
                    self.optimizer.step()

            loss_sum += float(loss.detach().item())
            steps += 1

            with torch.no_grad():
                for logits_v, target_v in views:
                    pred_v = torch.argmax(logits_v, dim=1)
                    valid = target_v != self.ignore_index
                    if valid.any():
                        dice_buf = accumulate_intersection_union(
                            pred_v[valid],
                            target_v[valid],
                            self.num_classes,
                            buffers=dice_buf,
                        )

            if (not training) and capture_vis and vis_payload is None and self.vis_enabled:
                n = min(self.vis_num, x.size(0))
                if n > 0:
                    vis_payload = {
                        "input": x[:n].detach(),
                        "target": y[:n].detach(),
                        "logits": views[0][0][:n].detach(),
                    }

        if steps == 0:
            raise RuntimeError(f"{desc} loader produced zero steps.")

        mean_loss = loss_sum / float(steps)
        if not math.isfinite(mean_loss):
            raise RuntimeError(f"Non-finite epoch loss in {desc.lower()} loop: {mean_loss}")

        per_class_dice, valid_mask = finalize_dice(
            dice_buf,
            eps=1e-6,
            empty_handling=self.dice_empty_handling,
        )
        macro = macro_dice(per_class_dice, valid_mask, include_bg=self.dice_include_bg)
        summary = dice_summary(per_class_dice, valid_mask, include_bg=self.dice_include_bg)

        return {
            "loss": float(mean_loss),
            "macro_dice": float(macro.item()) if torch.isfinite(macro) else float("nan"),
            "dice_min": float(summary["dice_min"].item()) if torch.isfinite(summary["dice_min"]) else float("nan"),
            "dice_mean": float(summary["dice_mean"].item()) if torch.isfinite(summary["dice_mean"]) else float("nan"),
            "dice_max": float(summary["dice_max"].item()) if torch.isfinite(summary["dice_max"]) else float("nan"),
            "num_valid_classes": int(summary["num_valid_classes"]),
            "per_class_dice": per_class_dice.detach().cpu(),
            "vis_payload": vis_payload,
        }

    def _save_latest(self, epoch: int) -> None:
        save_checkpoint(
            path=self.ckpt_dir / "latest.pt",
            epoch=epoch,
            best_val=self.best_macro_dice,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            cfg=self.cfg,
        )

    def _save_best(self, epoch: int) -> None:
        save_checkpoint(
            path=self.ckpt_dir / "best_eval_macro_dice.pt",
            epoch=epoch,
            best_val=self.best_macro_dice,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            cfg=self.cfg,
        )

    def fit(self, train_loader: DataLoader, eval_loader: DataLoader, epochs: int) -> None:
        epochs = int(epochs)
        if epochs != self._scheduler_total_epochs:
            self._build_scheduler(epochs)

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            if hasattr(self.model, "current_epoch"):
                self.model.current_epoch = epoch

            train_start = time.perf_counter()
            train_stats = self._run_epoch(train_loader, training=True)
            train_time = time.perf_counter() - train_start

            should_eval = (epoch % self.val_every == 0) or (epoch == epochs)
            capture_vis = should_eval and self.vis_enabled and (epoch % self.vis_every == 0)
            if should_eval:
                eval_start = time.perf_counter()
                eval_stats = self._run_epoch(eval_loader, training=False, capture_vis=capture_vis)
                eval_time = time.perf_counter() - eval_start
                self._val_vis_batch = eval_stats.get("vis_payload")
            else:
                eval_stats = {
                    "loss": float("nan"),
                    "macro_dice": float("nan"),
                    "dice_min": float("nan"),
                    "dice_mean": float("nan"),
                    "dice_max": float("nan"),
                    "num_valid_classes": 0,
                    "per_class_dice": None,
                    "vis_payload": None,
                }
                eval_time = 0.0
                self._val_vis_batch = None

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.logger.append(
                {
                    "epoch": epoch,
                    "train_loss": train_stats["loss"],
                    "train_macro_dice": train_stats["macro_dice"],
                    "train_dice_min": train_stats["dice_min"],
                    "train_dice_mean": train_stats["dice_mean"],
                    "train_dice_max": train_stats["dice_max"],
                    "train_num_valid_classes": train_stats["num_valid_classes"],
                    "eval_loss": eval_stats["loss"],
                    "eval_macro_dice": eval_stats["macro_dice"],
                    "eval_dice_min": eval_stats["dice_min"],
                    "eval_dice_mean": eval_stats["dice_mean"],
                    "eval_dice_max": eval_stats["dice_max"],
                    "eval_num_valid_classes": eval_stats["num_valid_classes"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

            if (epoch % self.save_latest_every == 0) or (epoch == epochs):
                self._save_latest(epoch)

            if should_eval:
                metric = float(eval_stats["macro_dice"])
                allow_best_window = (
                    epoch >= self.save_best_after_epoch
                    and ((epoch % self.save_best_every) == 0)
                )
                if allow_best_window and math.isfinite(metric) and metric > self.best_macro_dice:
                    self.best_macro_dice = metric
                    self._save_best(epoch)

            epoch_time = time.perf_counter() - epoch_start

            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                f"train loss={float(train_stats['loss']):.4f} "
                f"macro={float(train_stats['macro_dice']):.4f} "
                f"(min={float(train_stats['dice_min']):.4f}, mean={float(train_stats['dice_mean']):.4f}, "
                f"max={float(train_stats['dice_max']):.4f}, n={int(train_stats['num_valid_classes'])}) | "
                + (
                    (
                        f"eval loss={float(eval_stats['loss']):.4f} "
                        f"macro={float(eval_stats['macro_dice']):.4f} "
                        f"(min={float(eval_stats['dice_min']):.4f}, mean={float(eval_stats['dice_mean']):.4f}, "
                        f"max={float(eval_stats['dice_max']):.4f}, n={int(eval_stats['num_valid_classes'])}) | "
                        f"time=train:{train_time:.2f}s,eval:{eval_time:.2f}s,total:{epoch_time:.2f}s"
                    )
                    if should_eval
                    else f"eval skipped (val_every={self.val_every}) | time=train:{train_time:.2f}s,total:{epoch_time:.2f}s"
                )
            )

            preview_ids = range(min(self.num_classes, 16))
            train_cls_line = format_class_dice_line(train_stats["per_class_dice"], class_ids=preview_ids)
            train_suffix = " (preview first 16 classes)" if self.num_classes > 16 else ""
            print(f"[dice/train] {train_cls_line}{train_suffix}")
            if should_eval and eval_stats["per_class_dice"] is not None:
                eval_cls_line = format_class_dice_line(eval_stats["per_class_dice"], class_ids=preview_ids)
                eval_suffix = " (preview first 16 classes)" if self.num_classes > 16 else ""
                print(f"[dice/eval] {eval_cls_line}{eval_suffix}")

            if capture_vis and self.vis_dir is not None and self._val_vis_batch is not None:
                vis_path = self.vis_dir / f"eval_vis_epoch_{epoch:04d}.png"
                save_val_visualization_grid(
                    val_batch=self._val_vis_batch,
                    out_path=vis_path,
                    num_classes=self.num_classes,
                    class_names=self.class_names,
                    max_items=self.vis_num,
                )
                print(f"[vis] Saved eval visualization: {vis_path}")


__all__ = ["TissueSegmentationTrainer"]
