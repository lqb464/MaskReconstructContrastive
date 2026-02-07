from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import math

from ..config.experiment import ExperimentConfig
from ..models.swin_unet_dualview_ssl import SwinUNetDualViewSSL, flip_lr
from ..training.ckpt_io import save_checkpoint
from ..training.utils import ensure_dir
from tqdm import tqdm

from .dice import dice_coefficient
from .visualization import save_val_visualization_grid


class EpochLogger:
    """CSV logger with reconstruction + dice metrics."""

    HEADERS = [
        "epoch",
        "train_loss_total",
        "train_dice",
        "val_loss_total",
        "val_dice",
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
        self.vis_enabled = self.vis_every > 0 and self.vis_num > 0

        self.use_amp = bool(cfg.training.amp) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # cosine schedule with warmup
        total_epochs = int(cfg.training.epochs)
        warmup = int(getattr(cfg.training, "warmup_epochs", 0))
        min_lr = float(getattr(cfg.training, "min_lr", 0.0))
        base_lr = float(cfg.training.lr)

        def lr_lambda(epoch: int):
            if warmup > 0 and epoch < warmup:
                return float(epoch + 1) / float(warmup)
            t = epoch - warmup
            T = max(1, total_epochs - warmup)
            cosine = 0.5 * (1 + math.cos(math.pi * t / T))
            return (min_lr / base_lr) + (1 - min_lr / base_lr) * cosine

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        self.out_dir = ensure_dir(Path(out_dir))
        self.ckpt_dir = ensure_dir(self.out_dir / "checkpoints")
        self.vis_dir = ensure_dir(self.out_dir / "vis") if self.vis_enabled else None
        self.logger = EpochLogger(self.out_dir / "epoch_log.csv")

        self.best_val = float("-inf")
        self._val_vis_batch: dict[str, torch.Tensor] | None = None

    def _forward_losses(self, x: torch.Tensor, y: torch.Tensor, plane_one_hot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pixel_mask = torch.zeros_like(y)
        recon1, recon2, _, _ = self.model(x, pixel_mask, plane_one_hot)
        target_view2 = flip_lr(y) if (recon2 is not None and self.align_flip_target) else y

        loss_recon = F.binary_cross_entropy_with_logits(recon1, y)
        if recon2 is not None:
            loss_recon = 0.5 * (loss_recon + F.binary_cross_entropy_with_logits(recon2, target_view2))

        dice1 = dice_coefficient(torch.sigmoid(recon1), y, threshold=self.threshold)
        if recon2 is not None:
            dice2 = dice_coefficient(torch.sigmoid(recon2), target_view2, threshold=self.threshold)
            dice = 0.5 * (dice1 + dice2)
        else:
            dice = dice1

        lambda_recon = self.cfg.training.lambda_recon if self.cfg.training.lambda_recon > 0 else 1.0
        total = lambda_recon * loss_recon
        return total, dice

    def train_one_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        steps = 0
        progress = loader if self.disable_tqdm else tqdm(loader, desc="Train", leave=False, dynamic_ncols=True)

        for batch in progress:
            x = batch["input"].to(self.device, non_blocking=True)
            y = batch["target"].to(self.device, non_blocking=True)
            plane = batch["plane_one_hot"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, dice = self._forward_losses(x, y, plane)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.training.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.training.grad_clip)
                self.optimizer.step()

            total_loss += loss.detach().item()
            total_dice += dice.detach().item()
            steps += 1

            if not self.disable_tqdm:
                lr = self.optimizer.param_groups[0]["lr"]
                progress.set_postfix(
                    {
                        "lt": f"{total_loss/steps:.4f}",
                        "d": f"{total_dice/steps:.4f}",
                        "lr": f"{lr:.2e}",
                    }
                )

        if steps == 0:
            return 0.0, 0.0
        return (
            total_loss / steps,
            total_dice / steps,
        )

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        steps = 0
        self._val_vis_batch = None
        progress = loader if self.disable_tqdm else tqdm(loader, desc="Val", leave=False, dynamic_ncols=True)

        for batch in progress:
            x = batch["input"].to(self.device, non_blocking=True)
            y = batch["target"].to(self.device, non_blocking=True)
            plane = batch["plane_one_hot"].to(self.device, non_blocking=True)

            if steps == 0 and self.vis_enabled:
                n_vis = min(self.vis_num, x.size(0))
                if n_vis > 0:
                    self._val_vis_batch = {
                        "input": x[:n_vis].detach(),
                        "target": y[:n_vis].detach(),
                        "plane_one_hot": plane[:n_vis].detach(),
                    }

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, dice = self._forward_losses(x, y, plane)

            total_loss += loss.detach().item()
            total_dice += dice.detach().item()
            steps += 1

            if not self.disable_tqdm:
                progress.set_postfix(
                    {
                        "lt": f"{total_loss/steps:.4f}",
                        "d": f"{total_dice/steps:.4f}",
                    }
                )

        if steps == 0:
            return 0.0, 0.0
        return (
            total_loss / steps,
            total_dice / steps,
        )

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
        for epoch in range(1, epochs + 1):
            # Inform the model about epoch for any internal scheduling (e.g., SACA warmup)
            if hasattr(self.model, "current_epoch"):
                self.model.current_epoch = epoch

            train_loss, train_dice = self.train_one_epoch(train_loader)
            val_loss, val_dice = self.validate(val_loader)

            if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.logger.append(
                {
                    "epoch": epoch,
                    "train_loss_total": train_loss,
                    "train_dice": train_dice,
                    "val_loss_total": val_loss,
                    "val_dice": val_dice,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

            if not self.save_best_only:
                self._save_ckpt(epoch)

            # Selection rule: reconstruct -> maximize val_dice; else minimize val_loss
            metric = val_dice  # always select by val_dice
            if metric > self.best_val:
                self.best_val = metric
                self._save_best(epoch)

            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                f"train lt={train_loss:.4f} d={train_dice:.4f} | "
                f"val lt={val_loss:.4f} d={val_dice:.4f}"
            )

            if (
                self.vis_enabled
                and self.vis_dir is not None
                and val_loader is not None
                and self._val_vis_batch is not None
                and (epoch % self.vis_every == 0)
            ):
                vis_path = self.vis_dir / f"val_vis_epoch_{epoch:04d}.png"
                save_val_visualization_grid(
                    model=self.model,
                    val_batch=self._val_vis_batch,
                    device=self.device,
                    out_path=vis_path,
                    threshold=self.vis_threshold,
                    max_items=self.vis_num,
                    dice_fn=lambda p, t: dice_coefficient(p, t, threshold=self.vis_threshold),
                )
                print(f"[vis] Saved val visualization: {vis_path}")

            # scheduler already stepped; nothing else to do here


__all__ = ["MaskReconstructionTrainer"]
