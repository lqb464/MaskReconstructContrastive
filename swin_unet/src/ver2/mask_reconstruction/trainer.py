from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple, Callable

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import math

from ..config.experiment import ExperimentConfig
from ..models.swin_unet_dualview_ssl import SwinUNetDualViewSSL, flip_lr
from ..training.ckpt_io import save_checkpoint
from ..training.utils import ensure_dir
from tqdm import tqdm

from .dice import dice_coefficient, soft_dice_loss, soft_dice_loss_by_region
from .visualization import save_val_visualization_grid
from ..common.losses import nt_xent_loss, vicreg_loss, mixed_bce_logits_weighted_seg


class EpochLogger:
    """CSV logger with reconstruction + dice metrics."""

    HEADERS = [
        "epoch",
        "train_loss_total",
        "train_loss_masked",
        "train_loss_unmasked",
        "train_dice",
        "val_loss_total",
        "val_loss_masked",
        "val_loss_unmasked",
        "val_dice",
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
        self.vis_num = int(vis_num)
        self.vis_threshold = float(vis_threshold)
        self.disable_tqdm = bool(disable_tqdm)
        self.dice_loss_weight = float(cfg.training.dice_loss_weight)
        self.dice_mode = getattr(cfg.training, "dice_mode", "total")
        self.dice_smooth = getattr(cfg.training, "dice_smooth", 1e-6)

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
        self.vis_dir = ensure_dir(self.out_dir / "vis") if self.vis_every > 0 else None
        self.logger = EpochLogger(self.out_dir / "epoch_log.csv")

        self.best_val = float("-inf")

    def _forward_losses(self, x: torch.Tensor, y: torch.Tensor, plane_one_hot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pixel_mask = torch.zeros_like(y)  # No masking for this supervised task

        recon1, recon2, z1, z2 = self.model(x, pixel_mask, plane_one_hot)
        target_view2 = flip_lr(y) if (recon2 is not None and self.align_flip_target) else y

        # Reconstruction losses (mixed) and Dice metric/aux
        dice = torch.tensor(0.0, device=x.device)
        loss_total = torch.tensor(0.0, device=x.device)
        loss_masked = torch.tensor(0.0, device=x.device)
        loss_unmasked = torch.tensor(0.0, device=x.device)
        loss_dice_aux = torch.tensor(0.0, device=x.device)
        if self.cfg.training.enable_reconstruct:
            fg1 = (y > 0.5).float()
            lt1, lm1, lu1 = mixed_bce_logits_weighted_seg(
                recon1,
                y,
                fg1,
                fg_eps=self.cfg.training.fg_eps,
                fg_weight=self.cfg.training.fg_weight,
                alpha_mask=1.0,
                beta_unmask=0.2,
            )
            loss_total = loss_total + lt1
            loss_masked = loss_masked + lm1
            loss_unmasked = loss_unmasked + lu1

            if self.dice_loss_weight > 0:
                if self.dice_mode == "fg":
                    _, lmask1_dice, _ = soft_dice_loss_by_region(recon1, y, eps=self.dice_smooth)
                    loss_dice_aux = loss_dice_aux + lmask1_dice
                else:
                    ltot1_dice = soft_dice_loss(recon1, y, eps=self.dice_smooth)
                    loss_dice_aux = loss_dice_aux + ltot1_dice

            if recon2 is not None:
                fg2 = (target_view2 > 0.5).float()
                lt2, lm2, lu2 = mixed_bce_logits_weighted_seg(
                    recon2,
                    target_view2,
                    fg2,
                    fg_eps=self.cfg.training.fg_eps,
                    fg_weight=self.cfg.training.fg_weight,
                    alpha_mask=1.0,
                    beta_unmask=0.2,
                )
                loss_total = loss_total + lt2
                loss_masked = loss_masked + lm2
                loss_unmasked = loss_unmasked + lu2

                if self.dice_loss_weight > 0:
                    if self.dice_mode == "fg":
                        _, lmask2_dice, _ = soft_dice_loss_by_region(recon2, target_view2, eps=self.dice_smooth)
                        loss_dice_aux = loss_dice_aux + lmask2_dice
                    else:
                        ltot2_dice = soft_dice_loss(recon2, target_view2, eps=self.dice_smooth)
                        loss_dice_aux = loss_dice_aux + ltot2_dice

                dice2 = dice_coefficient(torch.sigmoid(recon2), target_view2, threshold=self.threshold)
            else:
                dice2 = torch.tensor(0.0, device=x.device)

            dice1 = dice_coefficient(torch.sigmoid(recon1), y, threshold=self.threshold)
            dice = 0.5 * (dice1 + dice2) if recon2 is not None else dice1

        # Contrastive
        loss_con = torch.tensor(0.0, device=x.device)
        if self.cfg.training.enable_contrastive:
            if z1 is None or z2 is None:
                raise RuntimeError("Contrastive enabled but model did not return embeddings.")
            if self.cfg.contrast_loss.contrastive_loss_type == "vicreg":
                loss_con = vicreg_loss(
                    z1=z1,
                    z2=z2,
                    invariance_weight=self.cfg.contrast_loss.vicreg_invariance_weight,
                    variance_weight=self.cfg.contrast_loss.vicreg_variance_weight,
                    covariance_weight=self.cfg.contrast_loss.vicreg_covariance_weight,
                    variance_eps=self.cfg.contrast_loss.vicreg_variance_eps,
                    target_std=self.cfg.contrast_loss.vicreg_target_std,
                )
            else:
                loss_con = nt_xent_loss(z1, z2, temperature=float(self.cfg.training.temperature))

        # Total
        lambda_recon = self.cfg.training.lambda_recon if self.cfg.training.lambda_recon > 0 else 1.0
        lambda_con = self.cfg.training.lambda_contrast
        total = torch.tensor(0.0, device=x.device)
        if self.cfg.training.enable_reconstruct:
            total = total + lambda_recon * loss_total
            if self.dice_loss_weight > 0:
                total = total + self.dice_loss_weight * loss_dice_aux
        if self.cfg.training.enable_contrastive and lambda_con != 0:
            total = total + lambda_con * loss_con
        if total.numel() == 0:
            total = loss_total + lambda_con * loss_con

        return total, dice, loss_con, loss_masked, loss_unmasked

    def train_one_epoch(self, loader: DataLoader) -> Tuple[float, float, float, float, float]:
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_con = 0.0
        total_masked = 0.0
        total_unmasked = 0.0
        steps = 0
        progress = loader if self.disable_tqdm else tqdm(loader, desc="Train", leave=False, dynamic_ncols=True)

        for batch in progress:
            x = batch["input"].to(self.device, non_blocking=True)
            y = batch["target"].to(self.device, non_blocking=True)
            plane = batch["plane_one_hot"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, dice, loss_con, loss_masked, loss_unmasked = self._forward_losses(x, y, plane)

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
            total_con += loss_con.detach().item()
            total_masked += loss_masked.detach().item()
            total_unmasked += loss_unmasked.detach().item()
            steps += 1

            if not self.disable_tqdm:
                lr = self.optimizer.param_groups[0]["lr"]
                progress.set_postfix(
                    {
                        "lt": f"{total_loss/steps:.4f}",
                        "lm": f"{total_masked/steps:.4f}",
                        "lu": f"{total_unmasked/steps:.4f}",
                        "d": f"{total_dice/steps:.4f}",
                        "lr": f"{lr:.2e}",
                    }
                )

        if steps == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        return (
            total_loss / steps,
            total_dice / steps,
            total_con / steps,
            total_masked / steps,
            total_unmasked / steps,
        )

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Tuple[float, float, float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_con = 0.0
        total_masked = 0.0
        total_unmasked = 0.0
        steps = 0
        progress = loader if self.disable_tqdm else tqdm(loader, desc="Val", leave=False, dynamic_ncols=True)

        for batch in progress:
            x = batch["input"].to(self.device, non_blocking=True)
            y = batch["target"].to(self.device, non_blocking=True)
            plane = batch["plane_one_hot"].to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, dice, loss_con, loss_masked, loss_unmasked = self._forward_losses(x, y, plane)

            total_loss += loss.detach().item()
            total_dice += dice.detach().item()
            total_con += loss_con.detach().item()
            total_masked += loss_masked.detach().item()
            total_unmasked += loss_unmasked.detach().item()
            steps += 1

            if not self.disable_tqdm:
                progress.set_postfix(
                    {
                        "lt": f"{total_loss/steps:.4f}",
                        "lm": f"{total_masked/steps:.4f}",
                        "lu": f"{total_unmasked/steps:.4f}",
                        "d": f"{total_dice/steps:.4f}",
                    }
                )

        if steps == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        return (
            total_loss / steps,
            total_dice / steps,
            total_con / steps,
            total_masked / steps,
            total_unmasked / steps,
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

            train_loss, train_dice, train_con, train_masked, train_unmasked = self.train_one_epoch(train_loader)
            val_loss, val_dice, val_con, val_masked, val_unmasked = self.validate(val_loader)

            if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.logger.append(
                {
                    "epoch": epoch,
                    "train_loss_total": train_loss,
                    "train_loss_masked": train_masked,
                    "train_loss_unmasked": train_unmasked,
                    "train_dice": train_dice,
                    "val_loss_total": val_loss,
                    "val_loss_masked": val_masked,
                    "val_loss_unmasked": val_unmasked,
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
                f"train lt={train_loss:.4f} lm={train_masked:.4f} lu={train_unmasked:.4f} d={train_dice:.4f} | "
                f"val lt={val_loss:.4f} lm={val_masked:.4f} lu={val_unmasked:.4f} d={val_dice:.4f}"
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

            # scheduler already stepped; nothing else to do here


__all__ = ["MaskReconstructionTrainer"]
