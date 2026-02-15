from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config.experiment import ExperimentConfig
from ..models.swin_unet_dualview_ssl import SwinUNetDualViewSSL, flip_lr
from ..training.utils import ensure_dir
from .ckpt_io import save_checkpoint
from .dice import dice_coefficient
from .visualization import save_val_visualization_grid

log = logging.getLogger(__name__)


class EpochLogger:
    """CSV logger with reconstruction + region-wise dice/loss metrics."""

    HEADERS = [
        "epoch",
        "train_loss_total",
        "train_loss_boundary",
        "train_loss_interior",
        "train_loss_dice_aux",
        "train_dice",
        "train_dice_boundary",
        "train_dice_interior",
        "val_loss_total",
        "val_loss_boundary",
        "val_loss_interior",
        "val_loss_dice_aux",
        "val_dice",
        "val_dice_boundary",
        "val_dice_interior",
        "lr",
    ]

    def __init__(self, path: Path):
        self.path = path
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(self.HEADERS)

    def append(self, row: Dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([row[h] for h in self.HEADERS])


class MaskReconstructionTrainer:
    def __init__(
        self,
        model: SwinUNetDualViewSSL,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        out_dir: Path,
        cfg: ExperimentConfig,
        *,
        threshold: float | None = None,
        save_best_only: bool = False,
        align_flip_target: bool = True,
        vis_every: int = 0,
        vis_num: int = 4,
        vis_threshold: float = 0.5,
        disable_tqdm: bool = False,
        train_step_dice: bool = False,
        boundary_aware: bool = False,
        val_every: int = 1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.threshold = threshold
        self.save_best_only = bool(save_best_only)
        self.align_flip_target = align_flip_target
        self.cfg = cfg
        self.vis_every = int(vis_every)
        cfg_logging = getattr(cfg, "logging", None)
        cfg_vis_num = int(getattr(cfg_logging, "vis_n_results", vis_num))
        self.vis_num = max(0, min(4, cfg_vis_num))
        self.vis_threshold = float(vis_threshold)
        self.disable_tqdm = bool(disable_tqdm)
        self.val_every = max(1, int(val_every))
        self.vis_enabled = self.vis_every > 0 and self.vis_num > 0
        cfg_train_step_dice = bool(getattr(cfg_logging, "train_step_dice", False))
        requested_train_step_dice = bool(train_step_dice) or cfg_train_step_dice
        # Recon-only task guardrail: dice is validation-only in config, but trainer still tracks epoch train dice.
        self.train_step_dice = False
        if requested_train_step_dice:
            log.warning("train_step_dice is ignored in mask reconstruction trainer (validation-only metric).")

        if bool(getattr(cfg.training, "enable_contrastive", False)) or bool(getattr(model, "enable_contrastive", False)):
            raise ValueError("Mask reconstruction trainer is recon-only; enable_contrastive must be False.")
        if bool(getattr(cfg.mask, "enable_masking", False)):
            log.warning("enable_masking=True is ignored in mask reconstruction trainer (reconstruction-only path).")
            cfg.mask.enable_masking = False

        cfg_boundary_aware = bool(getattr(cfg.training, "boundary_aware", False))
        self.boundary_aware = bool(boundary_aware) or cfg_boundary_aware
        # Fixed boundary settings for this mode by design.
        self.boundary_weight = 3.0
        self.boundary_thickness = 1
        self.boundary_target_threshold = 0.5
        self.dice_loss_weight = max(0.0, float(getattr(cfg.training, "dice_loss_weight", 0.0)))
        self.dice_mode = str(getattr(cfg.training, "dice_mode", "fg")).lower()
        if self.dice_mode not in {"fg", "total"}:
            log.warning("Unsupported dice_mode=%s; falling back to 'fg'.", self.dice_mode)
            self.dice_mode = "fg"
        self.dice_smooth = float(getattr(cfg.training, "dice_smooth", 1e-6))

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

        self.best_val = float("-inf")
        self._val_vis_batch: dict[str, torch.Tensor] | None = None

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

    @staticmethod
    def _metric_keys() -> tuple[str, ...]:
        return (
            "loss_total",
            "loss_boundary",
            "loss_interior",
            "loss_dice_aux",
            "dice_total",
            "dice_boundary",
            "dice_interior",
        )

    @classmethod
    def _zero_metric_dict(cls, *, device: torch.device | None = None) -> dict[str, torch.Tensor | float]:
        keys = cls._metric_keys()
        if device is None:
            return {k: 0.0 for k in keys}
        return {k: torch.zeros((), device=device) for k in keys}

    @classmethod
    def _nan_metric_dict(cls) -> dict[str, float]:
        return {k: float("nan") for k in cls._metric_keys()}

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        numer = (values * weights).sum(dim=(1, 2, 3))
        denom = weights.sum(dim=(1, 2, 3)).clamp_min(eps)
        return (numer / denom).mean()

    def _boundary_regions_from_target(self, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fg = (target > self.boundary_target_threshold).float()
        if self.boundary_thickness <= 0:
            boundary_band = torch.zeros_like(fg)
        else:
            k = 2 * int(self.boundary_thickness) + 1
            dilated = F.max_pool2d(fg, kernel_size=k, stride=1, padding=self.boundary_thickness)
            eroded = 1.0 - F.max_pool2d(1.0 - fg, kernel_size=k, stride=1, padding=self.boundary_thickness)
            boundary_band = ((dilated - eroded) > 0).float()

        boundary_fg = boundary_band * fg
        interior_fg = (fg - boundary_fg).clamp_min(0.0)
        return boundary_band, boundary_fg, interior_fg

    def _safe_region_dice(self, prob: torch.Tensor, target: torch.Tensor, region_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pred = (prob >= self.threshold).float() if self.threshold is not None else prob
        region_mask = region_mask.float()

        pred_r = pred * region_mask
        tgt_r = target.float() * region_mask
        pred_sum = pred_r.flatten(1).sum(dim=1)
        tgt_sum = tgt_r.flatten(1).sum(dim=1)
        inter = (pred_r * tgt_r).flatten(1).sum(dim=1)

        dice = (2.0 * inter + eps) / (pred_sum + tgt_sum + eps)
        empty = tgt_sum <= 0
        dice = torch.where(empty, torch.where(pred_sum <= 0, torch.ones_like(dice), torch.zeros_like(dice)), dice)
        return dice.mean()

    def _soft_dice_loss(
        self,
        prob: torch.Tensor,
        target: torch.Tensor,
        *,
        region_mask: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        target = target.float()
        if region_mask is not None:
            region_mask = region_mask.float()
            prob = prob * region_mask
            target = target * region_mask

        pred_sum = prob.flatten(1).sum(dim=1)
        tgt_sum = target.flatten(1).sum(dim=1)
        inter = (prob * target).flatten(1).sum(dim=1)
        dice = (2.0 * inter + eps) / (pred_sum + tgt_sum + eps)
        empty = tgt_sum <= 0
        dice = torch.where(empty, torch.where(pred_sum <= 0, torch.ones_like(dice), torch.zeros_like(dice)), dice)
        return (1.0 - dice).mean()

    def _view_metrics(self, logits: torch.Tensor, target: torch.Tensor, *, compute_dice: bool) -> dict[str, torch.Tensor]:
        bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        boundary_band, boundary_fg, interior_fg = self._boundary_regions_from_target(target)
        prob = torch.sigmoid(logits)

        if self.boundary_aware:
            pixel_weights = torch.ones_like(bce_map) + (self.boundary_weight - 1.0) * boundary_band
            loss_total = self._weighted_mean(bce_map, pixel_weights)
        else:
            loss_total = bce_map.mean()

        loss_boundary = self._weighted_mean(bce_map, boundary_fg)
        loss_interior = self._weighted_mean(bce_map, interior_fg)
        if self.dice_loss_weight > 0.0:
            if self.dice_mode == "fg":
                loss_dice_aux = self._soft_dice_loss(
                    prob,
                    target,
                    region_mask=(target > self.boundary_target_threshold).float(),
                    eps=self.dice_smooth,
                )
            else:
                loss_dice_aux = self._soft_dice_loss(prob, target, region_mask=None, eps=self.dice_smooth)
        else:
            loss_dice_aux = torch.zeros((), device=logits.device)

        metrics = {
            "loss_total": loss_total,
            "loss_boundary": loss_boundary,
            "loss_interior": loss_interior,
            "loss_dice_aux": loss_dice_aux,
            "dice_total": torch.zeros((), device=logits.device),
            "dice_boundary": torch.zeros((), device=logits.device),
            "dice_interior": torch.zeros((), device=logits.device),
        }

        if compute_dice:
            with torch.no_grad():
                metrics["dice_total"] = dice_coefficient(prob, target, threshold=self.threshold)
                metrics["dice_boundary"] = self._safe_region_dice(prob, target, boundary_fg)
                metrics["dice_interior"] = self._safe_region_dice(prob, target, interior_fg)

        return metrics

    def _merge_metrics(
        self,
        metrics1: dict[str, torch.Tensor],
        metrics2: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        if metrics2 is None:
            return metrics1
        return {k: 0.5 * (metrics1[k] + metrics2[k]) for k in self._metric_keys()}

    def _forward_losses(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        plane_one_hot: torch.Tensor,
        pixel_mask: torch.Tensor | None,
        *,
        compute_dice: bool,
        return_vis: bool = False,
        vis_items: int = 0,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
        assert x.shape == y.shape, f"Input/target shape mismatch before loss: {tuple(x.shape)} vs {tuple(y.shape)}"
        assert plane_one_hot.shape[0] == x.shape[0], "plane_one_hot batch dimension must match input batch size"

        recon1, recon2, _, _ = self.model(x, pixel_mask, plane_one_hot)
        assert recon1.shape == y.shape, f"recon1/target shape mismatch: {tuple(recon1.shape)} vs {tuple(y.shape)}"
        target_view2 = flip_lr(y) if (recon2 is not None and self.align_flip_target) else y
        if recon2 is not None:
            assert recon2.shape == target_view2.shape, (
                f"recon2/target_view2 shape mismatch: {tuple(recon2.shape)} vs {tuple(target_view2.shape)}"
            )

        metrics1 = self._view_metrics(recon1, y, compute_dice=compute_dice)
        metrics2 = self._view_metrics(recon2, target_view2, compute_dice=compute_dice) if recon2 is not None else None
        metrics = self._merge_metrics(metrics1, metrics2)

        lambda_recon = self.cfg.training.lambda_recon if self.cfg.training.lambda_recon > 0 else 1.0
        total_recon = metrics["loss_total"] + self.dice_loss_weight * metrics["loss_dice_aux"]
        total = lambda_recon * total_recon
        metrics["loss_total"] = total
        metrics["loss_boundary"] = lambda_recon * metrics["loss_boundary"]
        metrics["loss_interior"] = lambda_recon * metrics["loss_interior"]
        metrics["loss_dice_aux"] = lambda_recon * self.dice_loss_weight * metrics["loss_dice_aux"]

        vis_payload: dict[str, torch.Tensor] | None = None
        if return_vis and vis_items > 0:
            n_vis = min(int(vis_items), x.size(0))
            vis_payload = {
                "input": x[:n_vis].detach(),
                "target": y[:n_vis].detach(),
                "plane_one_hot": plane_one_hot[:n_vis].detach(),
                "recon1_logits": recon1[:n_vis].detach(),
            }
            if bool(getattr(self.cfg.mask, "enable_masking", False)):
                if pixel_mask is None:
                    log.warning("enable_masking=True is ignored in visualization payload path (pixel_mask is None).")
                else:
                    vis_payload["pixel_mask"] = pixel_mask[:n_vis].detach()
            elif pixel_mask is not None:
                vis_payload["pixel_mask"] = pixel_mask[:n_vis].detach()
            if recon2 is not None:
                vis_payload["recon2_logits"] = recon2[:n_vis].detach()
                vis_payload["target_flip"] = target_view2[:n_vis].detach()

        return total, metrics, vis_payload

    def _sample_pixel_mask(self, x: torch.Tensor) -> torch.Tensor | None:
        del x
        return None

    def train_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        totals = self._zero_metric_dict(device=None)
        steps = 0
        progress = loader if self.disable_tqdm else tqdm(loader, desc="Train", leave=False, dynamic_ncols=True)

        for batch in progress:
            x = batch["input"].to(self.device, non_blocking=True)
            y = batch["target"].to(self.device, non_blocking=True)
            plane = batch["plane_one_hot"].to(self.device, non_blocking=True)
            pixel_mask = self._sample_pixel_mask(x)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, metrics, _ = self._forward_losses(
                    x,
                    y,
                    plane,
                    pixel_mask,
                    compute_dice=True,
                )

            if self.use_amp:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.training.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.training.grad_clip)
                self.optimizer.step()

            for k in self._metric_keys():
                totals[k] += float(metrics[k].detach().item())
            steps += 1

        if steps == 0:
            return self._zero_metric_dict(device=None)
        return {k: float(v) / float(steps) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader, *, capture_vis: bool = False) -> Dict[str, float]:
        self.model.eval()
        totals = self._zero_metric_dict(device=None)
        steps = 0
        self._val_vis_batch = None
        progress = loader if self.disable_tqdm else tqdm(loader, desc="Val", leave=False, dynamic_ncols=True)

        for batch in progress:
            x = batch["input"].to(self.device, non_blocking=True)
            y = batch["target"].to(self.device, non_blocking=True)
            plane = batch["plane_one_hot"].to(self.device, non_blocking=True)
            pixel_mask = self._sample_pixel_mask(x)

            want_vis = False
            n_vis = 0
            if capture_vis and steps == 0 and self.vis_enabled:
                n_vis = min(self.vis_num, x.size(0))
                want_vis = n_vis > 0

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                _, metrics, vis_payload = self._forward_losses(
                    x,
                    y,
                    plane,
                    pixel_mask,
                    compute_dice=True,
                    return_vis=want_vis,
                    vis_items=n_vis,
                )

            if vis_payload is not None:
                self._val_vis_batch = vis_payload

            for k in self._metric_keys():
                totals[k] += float(metrics[k].detach().item())
            steps += 1

        if steps == 0:
            return self._zero_metric_dict(device=None)
        return {k: float(v) / float(steps) for k, v in totals.items()}

    def _save_ckpt(self, epoch: int) -> None:
        latest_path = self.ckpt_dir / "latest.pt"
        save_checkpoint(
            path=latest_path,
            epoch=epoch,
            best_val=self.best_val,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            cfg=self.cfg,
        )

    def _save_best(self, epoch: int) -> None:
        best_path = self.ckpt_dir / "best_val_dice.pt"
        save_checkpoint(
            path=best_path,
            epoch=epoch,
            best_val=self.best_val,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            cfg=self.cfg,
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
        epochs = int(epochs)
        if epochs != self._scheduler_total_epochs:
            if int(self.cfg.training.epochs) != epochs:
                log.warning(
                    "cfg.training.epochs (%d) differs from fit epochs (%d); scheduler will use fit epochs.",
                    int(self.cfg.training.epochs),
                    epochs,
                )
            self._build_scheduler(epochs)

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            if hasattr(self.model, "current_epoch"):
                self.model.current_epoch = epoch

            train_start = time.perf_counter()
            train_stats = self.train_one_epoch(train_loader)
            train_time = time.perf_counter() - train_start
            should_validate = (epoch % self.val_every == 0) or (epoch == epochs)
            capture_vis = should_validate and self.vis_enabled and (epoch % self.vis_every == 0)
            if should_validate:
                val_start = time.perf_counter()
                val_stats = self.validate(val_loader, capture_vis=capture_vis)
                val_time = time.perf_counter() - val_start
            else:
                val_stats = self._nan_metric_dict()
                val_time = 0.0
            epoch_time = time.perf_counter() - epoch_start

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.logger.append(
                {
                    "epoch": epoch,
                    "train_loss_total": train_stats["loss_total"],
                    "train_loss_boundary": train_stats["loss_boundary"],
                    "train_loss_interior": train_stats["loss_interior"],
                    "train_loss_dice_aux": train_stats["loss_dice_aux"],
                    "train_dice": train_stats["dice_total"],
                    "train_dice_boundary": train_stats["dice_boundary"],
                    "train_dice_interior": train_stats["dice_interior"],
                    "val_loss_total": val_stats["loss_total"],
                    "val_loss_boundary": val_stats["loss_boundary"],
                    "val_loss_interior": val_stats["loss_interior"],
                    "val_loss_dice_aux": val_stats["loss_dice_aux"],
                    "val_dice": val_stats["dice_total"],
                    "val_dice_boundary": val_stats["dice_boundary"],
                    "val_dice_interior": val_stats["dice_interior"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

            if not self.save_best_only:
                self._save_ckpt(epoch)

            if should_validate:
                metric = val_stats["dice_total"]
                if metric > self.best_val:
                    self.best_val = metric
                    self._save_best(epoch)

            if should_validate:
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} | "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                    f"train lt={train_stats['loss_total']:.4f} "
                    f"(b={train_stats['loss_boundary']:.4f}, i={train_stats['loss_interior']:.4f}) "
                    f"da={train_stats['loss_dice_aux']:.4f} "
                    f"d={train_stats['dice_total']:.4f} "
                    f"(b={train_stats['dice_boundary']:.4f}, i={train_stats['dice_interior']:.4f}) | "
                    f"val lt={val_stats['loss_total']:.4f} "
                    f"(b={val_stats['loss_boundary']:.4f}, i={val_stats['loss_interior']:.4f}) "
                    f"da={val_stats['loss_dice_aux']:.4f} "
                    f"d={val_stats['dice_total']:.4f} "
                    f"(b={val_stats['dice_boundary']:.4f}, i={val_stats['dice_interior']:.4f}) | "
                    f"time=train:{train_time:.2f}s,val:{val_time:.2f}s,total:{epoch_time:.2f}s"
                )
            else:
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} | "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                    f"train lt={train_stats['loss_total']:.4f} "
                    f"(b={train_stats['loss_boundary']:.4f}, i={train_stats['loss_interior']:.4f}) "
                    f"da={train_stats['loss_dice_aux']:.4f} "
                    f"d={train_stats['dice_total']:.4f} "
                    f"(b={train_stats['dice_boundary']:.4f}, i={train_stats['dice_interior']:.4f}) | "
                    f"val skipped (val_every={self.val_every}) | "
                    f"time=train:{train_time:.2f}s,total:{epoch_time:.2f}s"
                )

            if (
                capture_vis
                and self.vis_dir is not None
                and self._val_vis_batch is not None
            ):
                vis_path = self.vis_dir / f"val_vis_epoch_{epoch:04d}.png"
                save_val_visualization_grid(
                    val_batch=self._val_vis_batch,
                    device=self.device,
                    out_path=vis_path,
                    threshold=self.vis_threshold,
                    max_items=self.vis_num,
                    dice_fn=lambda p, t: dice_coefficient(p, t, threshold=self.vis_threshold),
                )
                print(f"[vis] Saved val visualization: {vis_path}")


__all__ = ["MaskReconstructionTrainer"]
