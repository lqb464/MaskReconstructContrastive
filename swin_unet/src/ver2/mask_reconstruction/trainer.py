from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Callable

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from models.swin_unet_dualview_ssl import SwinUNetDualViewSSL, flip_lr
from training.ckpt_io import save_checkpoint
from training.utils import ensure_dir

from .dice import dice_coefficient, soft_dice_loss
from .visualization import save_val_visualization_grid


@dataclass
class RunConfig:
    data_dir: str
    out_dir: str
    epochs: int
    batch_size: int
    lr: float
    num_workers: int
    seed: int
    amp: bool
    mask_key: str | None
    threshold: float | None
    strict_pairs: bool
    save_best_only: bool
    vis_every: int = 0
    vis_num: int = 4
    vis_threshold: float = 0.5


class EpochLogger:
    """Minimal CSV logger for epoch-level metrics."""

    def __init__(self, path: Path):
        self.path = path
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice"])

    def append(self, row: Dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([row["epoch"], row["train_loss"], row["train_dice"], row["val_loss"], row["val_dice"]])


class MaskReconstructionTrainer:
    def __init__(
        self,
        model: SwinUNetDualViewSSL,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        out_dir: Path,
        run_cfg: RunConfig,
        *,
        threshold: float | None = None,
        save_best_only: bool = False,
        align_flip_target: bool = True,
        vis_every: int = 0,
        vis_num: int = 4,
        vis_threshold: float = 0.5,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.threshold = threshold
        self.save_best_only = bool(save_best_only)
        self.align_flip_target = align_flip_target
        self.run_cfg = run_cfg
        self.vis_every = int(vis_every)
        self.vis_num = int(vis_num)
        self.vis_threshold = float(vis_threshold)

        self.use_amp = bool(run_cfg.amp) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        self.out_dir = ensure_dir(Path(out_dir))
        self.ckpt_dir = ensure_dir(self.out_dir / "checkpoints")
        self.vis_dir = ensure_dir(self.out_dir / "vis") if self.vis_every > 0 else None
        self.logger = EpochLogger(self.out_dir / "epoch_log.csv")

        self.best_val = float("-inf")

    def _forward_loss_dice(self, x: torch.Tensor, y: torch.Tensor, plane_one_hot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pixel_mask = torch.zeros_like(y)  # No masking for this supervised task

        recon1, recon2, _, _ = self.model(x, pixel_mask, plane_one_hot)
        target_view2 = flip_lr(y) if (recon2 is not None and self.align_flip_target) else y

        loss1 = soft_dice_loss(recon1, y)
        if recon2 is not None:
            loss2 = soft_dice_loss(recon2, target_view2)
            loss = loss1 + loss2
        else:
            loss2 = None
            loss = loss1

        dice1 = dice_coefficient(torch.sigmoid(recon1), y, threshold=self.threshold)
        if recon2 is not None:
            dice2 = dice_coefficient(torch.sigmoid(recon2), target_view2, threshold=self.threshold)
            dice = 0.5 * (dice1 + dice2)
        else:
            dice = dice1

        return loss, dice

    def train_one_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        steps = 0

        for batch in loader:
            x = batch["input"].to(self.device, non_blocking=True)
            y = batch["target"].to(self.device, non_blocking=True)
            plane = batch["plane_one_hot"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, dice = self._forward_loss_dice(x, y, plane)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.detach().item()
            total_dice += dice.detach().item()
            steps += 1

        if steps == 0:
            return 0.0, 0.0
        return total_loss / steps, total_dice / steps

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        steps = 0

        for batch in loader:
            x = batch["input"].to(self.device, non_blocking=True)
            y = batch["target"].to(self.device, non_blocking=True)
            plane = batch["plane_one_hot"].to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, dice = self._forward_loss_dice(x, y, plane)

            total_loss += loss.detach().item()
            total_dice += dice.detach().item()
            steps += 1

        if steps == 0:
            return 0.0, 0.0
        return total_loss / steps, total_dice / steps

    def _save_ckpt(self, epoch: int) -> None:
        latest_path = self.ckpt_dir / "latest.pt"
        save_checkpoint(
            path=latest_path,
            epoch=epoch,
            best_val=self.best_val,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            cfg=self.run_cfg,
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
            cfg=self.run_cfg,
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
        for epoch in range(1, epochs + 1):
            # Inform the model about epoch for any internal scheduling (e.g., SACA warmup)
            if hasattr(self.model, "current_epoch"):
                self.model.current_epoch = epoch

            train_loss, train_dice = self.train_one_epoch(train_loader)
            val_loss, val_dice = self.validate(val_loader)

            self.logger.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_dice": train_dice,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                }
            )

            if not self.save_best_only:
                self._save_ckpt(epoch)

            if val_dice > self.best_val:
                self.best_val = val_dice
                self._save_best(epoch)

            print(
                f"[epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
                f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
                f"best_val_dice={self.best_val:.4f}"
            )

            if self.vis_every > 0 and val_loader is not None and (epoch == 0 or (epoch % self.vis_every) == 0):
                vis_path = self.vis_dir / f"val_vis_epoch_{epoch:04d}.png"
                save_val_visualization_grid(
                    model=self.model,
                    val_loader=val_loader,
                    device=self.device,
                    out_path=vis_path,
                    threshold=self.vis_threshold,
                    max_items=self.vis_num,
                    dice_fn=lambda p, t: dice_coefficient(p, t, threshold=self.vis_threshold),
                )
                print(f"[vis] Saved val visualization: {vis_path}")


__all__ = ["MaskReconstructionTrainer", "RunConfig"]
