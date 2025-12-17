
# =============================================
# File: trainer_phase1.py
# Phase 1 trainer for SwinUNet Dual View SSL
# - No preprocessing pipeline
# - Keeps sample_masks_anti_mirror() logic
# - Dataset: folder with subfolders (see data.py)
# - Optional CSV labels for t-SNE (only plot if enabled AND labels exist)
# - Keeps reconstruction visualization grid
# =============================================
from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm

from config import ExperimentConfig, build_argparser
from data import create_dataloaders_from_folder
from augmentation import sample_masks_anti_mirror, HalfAug
from losses_patched import masked_l1_loss, mixed_l1_loss, nt_xent_loss, compute_embedding_variance, ssim_index, masked_bce_logits_weighted, mixed_bce_logits_weighted
from metrics import MetricsAccumulator
from visualization import save_image_grid, plot_training_curves, run_tsne_visualization
from model import SwinUNetDualViewSSLPhase1


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cpu: bool) -> torch.device:
    if cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def has_labels_in_batch(batch: Dict) -> bool:
    if "label" not in batch:
        return False
    y = batch["label"]
    if y is None:
        return False
    if isinstance(y, torch.Tensor):
        return y.numel() > 0
    return True


# -------------------------
# Trainer
# -------------------------
class Phase1Trainer:
    def __init__(self, cfg: ExperimentConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        out_dir = Path(cfg.logging.out_dir)
        if cfg.logging.run_name:
            out_dir = out_dir / cfg.logging.run_name
        self.out_dir = ensure_dir(out_dir)

        self.ckpt_dir = ensure_dir(Path(cfg.logging.ckpt_dir) if cfg.logging.ckpt_dir else (self.out_dir / "checkpoints"))
        self.vis_dir = ensure_dir(self.out_dir / "vis")
        self.plots_dir = ensure_dir(self.out_dir / "plots")

        self.log_csv_path = self.out_dir / "epoch_log.csv"
        self._init_csv()

        # Model
        self.model = SwinUNetDualViewSSLPhase1(
            in_ch=cfg.model.in_ch,
            image_size=cfg.data.image_size,
            patch_size=cfg.model.patch_size,
            embed_dim=cfg.model.embed_dim,
            depths=cfg.model.depths,
            num_heads=cfg.model.num_heads,
            window_size=cfg.model.window_size,
            proj_dim=cfg.model.proj_dim,
            plane_inject_method=cfg.model.plane_inject_method,
        ).to(device)

        # Optimizer
        self.opt = AdamW(self.model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        self.scaler = GradScaler(enabled=(cfg.training.amp and self.device.type == "cuda"))

        # Augmentation for contrastive halves (optional, used outside model)
        self.half_aug = HalfAug(
            p_noise=cfg.training.aug_p_noise,
            p_jitter=cfg.training.aug_p_jitter,
            p_blur=cfg.training.aug_p_blur,
            noise_std=cfg.training.aug_noise_std,
            jitter_strength=cfg.training.aug_jitter_strength,
            blur_kernel=cfg.training.aug_blur_kernel,
        )

        # Store datamodule/dataset reference for t-SNE label mapping if needed (optional)
        self.data_module = None

    def _init_csv(self):
        if self.log_csv_path.exists():
            return
        with self.log_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch",
                "train_loss",
                "train_recon_total",
                "train_recon_masked",
                "train_recon_unmasked",
                "train_ssim",
                "train_loss_contrast",
                "train_embed_var",
                "val_loss",
                "val_recon_total",
                "val_recon_masked",
                "val_recon_unmasked",
                "val_ssim",
            ])

    def _append_csv(self, row: Dict):
        header = None
        if self.log_csv_path.exists():
            with self.log_csv_path.open("r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
        if not header:
            return
        with self.log_csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([row.get(k, "") for k in header])

    @torch.no_grad()
    def _visualize_recon(self, x: torch.Tensor, pixel_mask: torch.Tensor, recon: torch.Tensor, epoch: int, tag: str):
        # x, recon: [B,1,H,W]; pixel_mask: [B,1,H,W] where 1=masked
    
        recon_full = pixel_mask * recon + (1.0 - pixel_mask) * x
        resid = (x - recon_full).abs().clamp(0, 1)
        masked = x * (1.0 - pixel_mask)

        out_path = str(self.vis_dir / f"{tag}_epoch_{epoch:03d}.png")
        save_image_grid(
            [x, pixel_mask, masked, recon_full.clamp(0, 1), resid],
            [f"{tag}: target", "mask", "masked", "recon_full", "abs_resid"],
            out_path,
        )
        
        print("x     min/max/mean:", x.min().item(), x.max().item(), x.mean().item())
        print("recon min/max/mean:", recon.min().item(), recon.max().item(), recon.mean().item())
        print("mask  mean:", pixel_mask.float().mean().item())


    def train_one_epoch(self, loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        meter = MetricsAccumulator()
        losses = []
        losses_con = []
        embed_vars = []

        pbar = tqdm(loader, desc=f"Train {epoch}", leave=False)
        for step, batch in enumerate(pbar):
            x = batch["input"].to(self.device, non_blocking=True)  # [B,1,H,W]
            plane = batch.get("plane_one_hot", None)
            if plane is None:
                plane = torch.tensor([0.0, 1.0], device=self.device).view(1, 2).repeat(x.size(0), 1)
            else:
                plane = plane.to(self.device, non_blocking=True)

            pixel_mask = sample_masks_anti_mirror(x.size(0), self.cfg.mask, self.device)

            # --- Collapse diagnostics (run once per epoch on first batch) ---
            if step == 0:
                with torch.no_grad():
                    x0 = x.detach()
                    frac_black = float((x0 < 0.01).to(torch.float32).mean().item())
                    med = float(x0.median().item())
                    mean = float(x0.mean().item())
                    m = pixel_mask.detach()
                    masked_mean = float((x0 * m).sum().item() / m.sum().clamp(min=1.0).item())
                    unmasked_mean = float((x0 * (1.0 - m)).sum().item() / (1.0 - m).sum().clamp(min=1.0).item())
                    print(f"[diag e{epoch:03d}] x mean={mean:.6f} median={med:.6f} frac(x<0.01)={frac_black:.3f} masked_mean={masked_mean:.6f} unmasked_mean={unmasked_mean:.6f}")
            # ---------------------------------------------------------------


            self.opt.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=(self.cfg.training.amp and self.device.type == "cuda")):
                # recon_raw là logits (KHÔNG sigmoid trong model)
                recon_raw, z1, z2 = self.model(
                    x, pixel_mask=pixel_mask, plane_one_hot=plane, return_embeddings=True
                )
                # recon_raw is logits (NO sigmoid in the model)
                if step == 0:
                    print("recon_raw min/max/mean:", recon_raw.min().item(), recon_raw.max().item(), recon_raw.mean().item())

                # For metrics/visualization we keep a bounded image in [0,1]
                recon_img = torch.sigmoid(recon_raw.clamp(-10, 10))
                if step == 0:
                    print("recon_img min/max/mean:", recon_img.min().item(), recon_img.max().item(), recon_img.mean().item())

                # Reconstruction loss: avoid sigmoid saturation + all-zero collapse on sparse images
                recon_loss_type = getattr(self.cfg.training, "recon_loss", "weighted_bce_logits")
                fg_eps = float(getattr(self.cfg.training, "fg_eps", 0.02))
                fg_weight = float(getattr(self.cfg.training, "fg_weight", 10.0))

                if recon_loss_type == "weighted_bce_logits":
                    if self.cfg.training.enable_masked_loss:
                        loss_recon = masked_bce_logits_weighted(recon_raw, x, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight)
                    else:
                        loss_recon = mixed_bce_logits_weighted(recon_raw, x, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight)
                else:
                    # legacy: L1 on sigmoid output
                    if self.cfg.training.enable_masked_loss:
                        loss_recon = masked_l1_loss(recon_img, x, pixel_mask)
                    else:
                        loss_recon = mixed_l1_loss(recon_img, x, pixel_mask)

                if self.cfg.training.enable_contrastive:
                    loss_con = nt_xent_loss(z1, z2, temperature=self.cfg.training.temperature)
                else:
                    loss_con = torch.zeros((), device=self.device)

                loss = self.cfg.training.lambda_recon * loss_recon + self.cfg.training.lambda_contrast * loss_con

            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

            with torch.no_grad():
                diff = (x - recon_img).abs().detach()

                ssim_vals = ssim_index(x.float(), recon_img.float())  # shape (B,)
                ssim_sum = float(ssim_vals.sum().item())
                meter.update(diff, pixel_mask, ssim_sum=ssim_sum)

                losses.append(float(loss.item()))
                losses_con.append(float(loss_con.item()) if torch.is_tensor(loss_con) else 0.0)

                if self.cfg.training.enable_contrastive:
                    mean_var, min_var = compute_embedding_variance([z1.detach(), z2.detach()])
                    embed_vars.append(float(mean_var))
                else:
                    embed_vars.append(0.0)

            pbar.set_postfix(loss=np.mean(losses[-20:]) if losses else 0.0)

        stats = meter.compute()  # ReconstructionMetrics
        out = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "loss_contrast": float(np.mean(losses_con)) if losses_con else 0.0,
            "embed_var": float(np.mean(embed_vars)) if embed_vars else 0.0,
            "recon_total": float(stats.total_l1),
            "recon_masked": float(stats.masked_l1),
            "recon_unmasked": float(stats.unmasked_l1),
            "ssim": float(stats.ssim),
        }
        return out

    @torch.no_grad()
    def validate(self, loader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        meter = MetricsAccumulator()
        losses = []

        for batch in tqdm(loader, desc=f"val {epoch}", leave=False):
            x = batch["input"].to(self.device, non_blocking=True)
            plane = batch.get("plane_one_hot", None)
            if plane is None:
                plane = torch.tensor([0.0, 1.0], device=self.device).view(1, 2).repeat(x.size(0), 1)
            else:
                plane = plane.to(self.device, non_blocking=True)

            pixel_mask = sample_masks_anti_mirror(x.size(0), self.cfg.mask, self.device)

            with autocast(device_type=self.device.type, enabled=(self.cfg.training.amp and self.device.type == "cuda")):
                recon_raw, z1, z2 = self.model(
                    x, pixel_mask=pixel_mask, plane_one_hot=plane, return_embeddings=False
                )

                # For metrics/visualization we keep a bounded image in [0,1]
                recon_img = torch.sigmoid(recon_raw.clamp(-10, 10))

                recon_loss_type = getattr(self.cfg.training, "recon_loss", "weighted_bce_logits")
                fg_eps = float(getattr(self.cfg.training, "fg_eps", 0.02))
                fg_weight = float(getattr(self.cfg.training, "fg_weight", 10.0))

                if recon_loss_type == "weighted_bce_logits":
                    if self.cfg.training.enable_masked_loss:
                        loss_recon = masked_bce_logits_weighted(recon_raw, x, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight)
                    else:
                        loss_recon = mixed_bce_logits_weighted(recon_raw, x, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight)
                else:
                    if self.cfg.training.enable_masked_loss:
                        loss_recon = masked_l1_loss(recon_img, x, pixel_mask)
                    else:
                        loss_recon = mixed_l1_loss(recon_img, x, pixel_mask)

                loss = self.cfg.training.lambda_recon * loss_recon

            diff = (x - recon_img).abs()

            ssim_vals = ssim_index(x.float(), recon_img.float())  # shape (B,)
            ssim_sum = float(ssim_vals.sum().item())
            meter.update(diff, pixel_mask, ssim_sum=ssim_sum)

            losses.append(float(loss.item()))

        stats = meter.compute()  # ReconstructionMetrics
        out = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "recon_total": float(stats.total_l1),
            "recon_masked": float(stats.masked_l1),
            "recon_unmasked": float(stats.unmasked_l1),
            "ssim": float(stats.ssim),
        }
        return out

    def maybe_visualize(self, loader, epoch: int, tag: str):
        if (epoch % self.cfg.logging.vis_every) != 0:
            return
        self.model.eval()
        batch = next(iter(loader))
        x = batch["input"].to(self.device, non_blocking=True)
        plane = batch.get("plane_one_hot", None)
        if plane is None:
            plane = torch.tensor([0.0, 1.0], device=self.device).view(1, 2).repeat(x.size(0), 1)
        else:
            plane = plane.to(self.device, non_blocking=True)

        pixel_mask = sample_masks_anti_mirror(x.size(0), self.cfg.mask, self.device)

        recon_raw, _, _ = self.model(
            x,
            pixel_mask=pixel_mask,
            plane_one_hot=plane,
            return_embeddings=False
        )
        recon_img = torch.sigmoid(recon_raw.clamp(-10, 10))

        self._visualize_recon(x, pixel_mask, recon_img, epoch, tag)


    def maybe_tsne(self, loader, epoch: int):
        if not self.cfg.logging.enable_tsne:
            return
        if (epoch % self.cfg.logging.tsne_every) != 0:
            return

        # Gating: only if labels exist (unless user overrides)
        if self.cfg.logging.tsne_only_if_labeled:
            # check first batch for label presence
            try:
                b0 = next(iter(loader))
                if not has_labels_in_batch(b0):
                    return
            except Exception:
                return

        out_prefix = str(self.out_dir / "tsne" / f"epoch_{epoch:03d}")
        run_tsne_visualization(
            model=self._tsne_wrapper_model(),
            loader=loader,
            device=self.device,
            out_prefix=out_prefix,
            max_items=self.cfg.logging.tsne_max_items,
            label_val="label",
            data_module=self.data_module,
        )

    def _tsne_wrapper_model(self):
        """
        run_tsne_visualization expects model.encoder_embed(x, mode=...)
        Provide a light wrapper that uses view1 encoder bottleneck pooled.
        """
        trainer = self

        class _Wrap(nn.Module):
            def __init__(self, base: SwinUNetDualViewSSLPhase1):
                super().__init__()
                self.base = base

            @torch.no_grad()
            def encoder_embed(self, x: torch.Tensor, mode: str = "bottleneck"):
                # x: [B,1,H,W], no mask in tSNE; use zero mask
                B, _, H, W = x.shape
                device = x.device
                M = torch.zeros((B, 1, H, W), device=device, dtype=x.dtype)
                plane = torch.tensor([0.0, 1.0], device=device).view(1, 2).repeat(B, 1)
                b = self.base.encode_bottleneck(x, plane, view=1)  # [B,h,w,C]
                h = b.mean(dim=(1, 2))  # [B,C]
                return None, h

        return _Wrap(self.model).to(self.device)

    def save_checkpoint(self, epoch: int):
        path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
        obj = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "cfg": asdict(self.cfg),
        }
        torch.save(obj, path)

    def fit(self, train_loader, val_loader):
        best_val = float("inf")
        for epoch in range(1, self.cfg.training.epochs + 1):
            t0 = time.time()
            tr = self.train_one_epoch(train_loader, epoch)
            va = self.validate(val_loader, epoch)
            dt = time.time() - t0

            row = {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_recon_total": tr["recon_total"],
                "train_recon_masked": tr["recon_masked"],
                "train_recon_unmasked": tr["recon_unmasked"],
                "train_ssim": tr["ssim"],
                "train_loss_contrast": tr["loss_contrast"],
                "train_embed_var": tr["embed_var"],
                "val_loss": va["loss"],
                "val_recon_total": va["recon_total"],
                "val_recon_masked": va["recon_masked"],
                "val_recon_unmasked": va["recon_unmasked"],
                "val_ssim": va["ssim"],
            }
            self._append_csv(row)

            # hooks
            self.maybe_visualize(val_loader, epoch, tag="val")
            self.maybe_tsne(val_loader, epoch)

            # checkpoint
            if va["loss"] < best_val:
                best_val = va["loss"]
                self.save_checkpoint(epoch)

            # update plots
            plot_training_curves(self.log_csv_path, self.plots_dir)

            print(f"[epoch {epoch:03d}] train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} time={dt:.1f}s")


# -------------------------
# Main
# -------------------------
def main():
    parser = build_argparser()
    args = parser.parse_args()
    cfg = ExperimentConfig.from_args(args)

    set_seed(cfg.training.seed)
    device = get_device(cfg.training.cpu)

    train_loader, val_loader, test_loader, full_ds = create_dataloaders_from_folder(
        data_root=cfg.data.data_root,
        image_size=cfg.data.image_size,
        plane=cfg.data.plane,
        label_csv=cfg.data.label_csv if cfg.data.label_csv else None,
        label_path_col=cfg.data.label_path_col,
        label_col=cfg.data.label_col,
        batch_size=cfg.training.batch_size,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        seed=cfg.training.seed,
        drop_last=cfg.data.drop_last,
    )

    print(f"Dataset size: {len(full_ds)}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    trainer = Phase1Trainer(cfg, device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()