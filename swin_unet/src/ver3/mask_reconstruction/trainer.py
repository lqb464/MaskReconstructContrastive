from __future__ import annotations

import csv
import logging
import math
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
        train_step_dice: bool = False,
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
        cfg_train_step_dice = bool(getattr(cfg_logging, "train_step_dice", False))
        requested_train_step_dice = bool(train_step_dice) or cfg_train_step_dice
        # Recon-only task guardrail: dice is validation-only in this entrypoint.
        self.train_step_dice = False
        if requested_train_step_dice:
            log.warning("train_step_dice is ignored in mask reconstruction trainer (validation-only metric).")

        if bool(getattr(cfg.training, "enable_contrastive", False)) or bool(getattr(model, "enable_contrastive", False)):
            raise ValueError("Mask reconstruction trainer is recon-only; enable_contrastive must be False.")
        if bool(getattr(cfg.mask, "enable_masking", False)):
            raise ValueError("Mask reconstruction trainer expects masking disabled (cfg.mask.enable_masking=False).")

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

    def _forward_losses(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        plane_one_hot: torch.Tensor,
        *,
        compute_dice: bool,
        return_vis: bool = False,
        vis_items: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        assert x.shape == y.shape, f"Input/target shape mismatch before loss: {tuple(x.shape)} vs {tuple(y.shape)}"
        assert plane_one_hot.shape[0] == x.shape[0], "plane_one_hot batch dimension must match input batch size"

        # Recon-only task invariant: pixel masking is disabled, so pass None to avoid per-step mask allocation.
        pixel_mask = None
        assert pixel_mask is None, "pixel_mask must remain None in mask reconstruction trainer."
        recon1, recon2, _, _ = self.model(x, pixel_mask, plane_one_hot)
        assert recon1.shape == y.shape, f"recon1/target shape mismatch: {tuple(recon1.shape)} vs {tuple(y.shape)}"
        target_view2 = flip_lr(y) if (recon2 is not None and self.align_flip_target) else y
        if recon2 is not None:
            assert recon2.shape == target_view2.shape, (
                f"recon2/target_view2 shape mismatch: {tuple(recon2.shape)} vs {tuple(target_view2.shape)}"
            )

        loss_recon = F.binary_cross_entropy_with_logits(recon1, y)
        if recon2 is not None:
            loss_recon = 0.5 * (loss_recon + F.binary_cross_entropy_with_logits(recon2, target_view2))

        dice = torch.zeros((), device=x.device)
        if compute_dice:
            # Metric-only path: keep this under no_grad even if caller toggles compute_dice in training by mistake.
            with torch.no_grad():
                prob1 = torch.sigmoid(recon1)
                dice1 = dice_coefficient(prob1, y, threshold=self.threshold)
                if recon2 is not None:
                    prob2 = torch.sigmoid(recon2)
                    dice2 = dice_coefficient(prob2, target_view2, threshold=self.threshold)
                    dice = 0.5 * (dice1 + dice2)
                else:
                    dice = dice1

        lambda_recon = self.cfg.training.lambda_recon if self.cfg.training.lambda_recon > 0 else 1.0
        total = lambda_recon * loss_recon

        vis_payload: dict[str, torch.Tensor] | None = None
        if return_vis and vis_items > 0:
            n_vis = min(int(vis_items), x.size(0))
            vis_payload = {
                "input": x[:n_vis].detach(),
                "target": y[:n_vis].detach(),
                "plane_one_hot": plane_one_hot[:n_vis].detach(),
                "recon1_logits": recon1[:n_vis].detach(),
            }
            if recon2 is not None:
                vis_payload["recon2_logits"] = recon2[:n_vis].detach()
                vis_payload["target_flip"] = target_view2[:n_vis].detach()

        return total, dice, vis_payload

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
                loss, dice, _ = self._forward_losses(
                    x,
                    y,
                    plane,
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

            total_loss += loss.detach().item()
            total_dice += dice.detach().item()
            steps += 1

            if not self.disable_tqdm:
                progress.set_postfix(
                    {
                        "lt": f"{total_loss/steps:.4f}",
                    }
                )

        if steps == 0:
            return 0.0, 0.0
        return total_loss / steps, total_dice / steps

    @torch.no_grad()
    def validate(self, loader: DataLoader, *, capture_vis: bool = False) -> Tuple[float, float]:
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

            want_vis = False
            n_vis = 0
            if capture_vis and steps == 0 and self.vis_enabled:
                n_vis = min(self.vis_num, x.size(0))
                want_vis = n_vis > 0

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, dice, vis_payload = self._forward_losses(
                    x,
                    y,
                    plane,
                    compute_dice=True,
                    return_vis=want_vis,
                    vis_items=n_vis,
                )

            if vis_payload is not None:
                self._val_vis_batch = vis_payload

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
            if hasattr(self.model, "current_epoch"):
                self.model.current_epoch = epoch

            train_loss, train_dice = self.train_one_epoch(train_loader)
            capture_vis = self.vis_enabled and (epoch % self.vis_every == 0)
            val_loss, val_dice = self.validate(val_loader, capture_vis=capture_vis)

            if self.lr_scheduler is not None:
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

            metric = val_dice
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
