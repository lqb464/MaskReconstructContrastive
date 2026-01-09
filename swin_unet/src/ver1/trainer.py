
# =============================================
# File: trainer.py
# Phase A + Phase B
#
# Phase A: Dual Reconstruction Head (Original + Flip)
# - Keeps sample_masks_anti_mirror() logic unchanged
# - Keeps contrastive pairing and NT-Xent as-is
# - Adds second reconstruction loss for flipped target in same batch
# - Logs parameter counts: total, encoder, decoder_trunk, recon_heads
#
# Phase B: Explicit Loss Decomposition Logging
# - Logs per-epoch (and optional per-iteration via tqdm postfix):
#   * loss_recon_orig
#   * loss_recon_flip
#   * loss_recon_total
#   * loss_contrastive
#   * loss_total
# - Validation logs at least reconstruction decomposition (orig/flip/total)
# - Writes a separate CSV (loss_decomp.csv) with clear headers (no overwrite)
# =============================================
from __future__ import annotations

import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torchinfo import summary
from torch import nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm

from config import ExperimentConfig, build_argparser
from data import create_dataloaders_from_folder
from augmentation import sample_masks_anti_mirror
from losses import masked_l1_loss, mixed_l1_loss, nt_xent_loss, compute_embedding_variance, ssim_index
from metrics import MetricsAccumulator
from visualization import save_image_grid, plot_training_curves, run_tsne_visualization, plot_loss_decomposition_curves
from model import SwinUNetDualViewSSLPhase1, flip_lr


# -------------------------
# Weighted BCE logits (kept local to trainer)
# -------------------------
def _foreground_weighted_bce_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    fg_eps: float = 0.02,
    fg_weight: float = 10.0,
) -> torch.Tensor:
    """
    Weighted BCEWithLogits where pixels with target > fg_eps get larger weight.
    target expected in [0,1].
    """
    with torch.no_grad():
        w = torch.ones_like(target)
        w = torch.where(target > fg_eps, torch.full_like(w, fg_weight), w)
    return F.binary_cross_entropy_with_logits(logits, target, weight=w, reduction="none")


def masked_bce_logits_weighted(
    logits: torch.Tensor,
    target: torch.Tensor,
    pixel_mask: torch.Tensor,
    fg_eps: float = 0.02,
    fg_weight: float = 10.0,
) -> torch.Tensor:
    """
    BCE logits computed only on masked region (pixel_mask==1).
    """
    loss_map = _foreground_weighted_bce_logits(logits, target, fg_eps=fg_eps, fg_weight=fg_weight)
    m = pixel_mask
    denom = m.sum().clamp(min=1.0)
    return (loss_map * m).sum() / denom


def mixed_bce_logits_weighted(
    logits: torch.Tensor,
    target: torch.Tensor,
    pixel_mask: torch.Tensor,
    fg_eps: float = 0.02,
    fg_weight: float = 10.0,
    alpha_mask: float = 1.0,
    beta_unmask: float = 0.2,
) -> torch.Tensor:
    """
    Weighted BCE logits computed on both masked and unmasked, with different weights.
    """
    loss_map = _foreground_weighted_bce_logits(logits, target, fg_eps=fg_eps, fg_weight=fg_weight)
    m = pixel_mask
    um = 1.0 - m
    masked = (loss_map * m).sum() / m.sum().clamp(min=1.0)
    unmasked = (loss_map * um).sum() / um.sum().clamp(min=1.0)
    return alpha_mask * masked + beta_unmask * unmasked


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
    y = batch.get("label", None)
    return isinstance(y, torch.Tensor) and y.numel() > 0


# -------------------------
# Trainer
# -------------------------
class PhaseATrainer:
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
        self.tsne_dir = ensure_dir(self.out_dir / "tsne")

        # Epoch summary CSV (existing)
        self.log_csv_path = self.out_dir / "epoch_log.csv"
        self._init_epoch_csv()

        # Phase B: explicit decomposition CSV (new, no overwrite)
        self.loss_decomp_csv_path = self.out_dir / "loss_decomp.csv"
        self._init_loss_decomp_csv()

        self.model = SwinUNetDualViewSSLPhase1(
            in_ch=cfg.model.in_ch,
            image_size=cfg.data.image_size,
            patch_size=cfg.model.patch_size,
            embed_dim=cfg.model.embed_dim,
            enc_depths=tuple(cfg.model.enc_depths),
            dec_depths=tuple(cfg.model.dec_depths),
            num_heads=tuple(cfg.model.num_heads),
            window_size=cfg.model.window_size,
            proj_dim=cfg.model.proj_dim,
            plane_inject_method=cfg.model.plane_inject_method,
            enable_saca_stage1=cfg.model.enable_saca_stage1,
        ).to(device)
        
        print(self.model)

        # Parameter logging (Phase A DoD)
        try:
            pc = self.model.param_count_breakdown()

            print("[params] total:", pc.get("total", 0))

            print("[params] enc_early_view1:", pc.get("enc_early_view1", 0))
            print("[params] enc_early_view2:", pc.get("enc_early_view2", 0))
            print("[params] enc_shared_trunk:", pc.get("enc_shared_trunk", 0))

            print("[params] contrastive_head:", pc.get("contrastive_head", 0))

            print("[params] decoder_shared_up2:", pc.get("decoder_shared_up2", 0))
            print("[params] decoder_branch_v1:", pc.get("decoder_branch_v1", 0))
            print("[params] decoder_branch_v2:", pc.get("decoder_branch_v2", 0))

            print("[params] recon_heads:", pc.get("recon_heads", 0))

            print("[params] check_sum:", pc.get("check_sum", 0))
            print("[params] delta_total_minus_check:", pc.get("delta_total_minus_check", 0))

        except Exception as e:
            print("[params] unable to compute breakdown:", repr(e))
            
        try:
            B = cfg.train.batch_size if hasattr(cfg, "train") else 1
            H = cfg.data.image_size
            W = cfg.data.image_size
            in_ch = cfg.model.in_ch

            dummy_x = torch.zeros(B, in_ch, H, W, device=device)
            dummy_pixel_mask = torch.zeros(B, 1, H, W, device=device)
            dummy_plane_one_hot = torch.zeros(B, 2, device=device)

            print("\n[torchinfo] Model architecture summary\n")

            summary(
                self.model,
                input_data=(
                    dummy_x,
                    dummy_pixel_mask,
                    dummy_plane_one_hot,
                ),
                depth=4,                  # 3–5 là hợp lý, sâu hơn rất dài
                col_names=(
                    "input_size",
                    "output_size",
                    "num_params",
                    "trainable",
                ),
                verbose=1,
                device=device,
            )

        except Exception as e:
            print("[torchinfo] unable to print model summary:", repr(e))


        self.opt = AdamW(self.model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        self.scaler = GradScaler(enabled=(cfg.training.amp and device.type == "cuda"))

        # Optional: stored for TSNE tool compatibility
        self.data_module = None

    def _lambda_contrastive_eff(self, epoch: int) -> float:
        """
        Linear ramp for contrastive weight:
          epoch 1..ramp_epochs: scale from ~0 -> 1
          epoch > ramp_epochs: scale = 1
        If ramp is disabled (<=0), return base lambda_contrast.
       """
        base = float(getattr(self.cfg.training, "lambda_contrast", 0.0))
        # Accept a few possible config names to be robust
        ramp_epochs = int(
            getattr(
                self.cfg.training,
                "ramp_contrastive",
                getattr(self.cfg.training, "ramp_contrastive_epochs", 0),
            )
        )
        if ramp_epochs <= 0:
            return base
        scale = min(1.0, float(epoch) / float(ramp_epochs))
        return base * scale

    # -------- CSV init/append --------
    def _init_epoch_csv(self):
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
                "train_embed_var_mean",
                "train_embed_var_min",
                "val_loss",
                "val_recon_total",
                "val_recon_masked",
                "val_recon_unmasked",
                "val_ssim",
            ])

    def _append_epoch_csv(self, row: Dict):
        with self.log_csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                row["epoch"],
                row["train_loss"],
                row["train_recon_total"],
                row["train_recon_masked"],
                row["train_recon_unmasked"],
                row["train_ssim"],
                row["train_loss_contrast"],
                row["train_embed_var_mean"],
                row["train_embed_var_min"],
                row["val_loss"],
                row["val_recon_total"],
                row["val_recon_masked"],
                row["val_recon_unmasked"],
                row["val_ssim"],
            ])

    def _init_loss_decomp_csv(self):
        if self.loss_decomp_csv_path.exists():
            return
        with self.loss_decomp_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch",
                "split",  # train/val
                "loss_recon_orig",
                "loss_recon_flip",
                "loss_recon_total",
                "loss_contrastive",
                "loss_total",
            ])

    def _append_loss_decomp_csv(self, epoch: int, split: str, d: Dict[str, float]):
        with self.loss_decomp_csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                epoch,
                split,
                d.get("loss_recon_orig", 0.0),
                d.get("loss_recon_flip", 0.0),
                d.get("loss_recon_total", 0.0),
                d.get("loss_contrastive", 0.0),
                d.get("loss_total", 0.0),
            ])

    # -------- Visualization (kept: recon on orig) --------
    @torch.no_grad()
    def _visualize_recon(self, target_view: torch.Tensor, pixel_mask: torch.Tensor, recon_img: torch.Tensor, epoch: int, tag: str):
        masked_in = target_view * (1.0 - pixel_mask)

        if self.cfg.training.enable_masked_loss:
            shown_target = (1.0 - pixel_mask) * target_view + pixel_mask * recon_img
            shown_title = f"{tag}: target(unmask)+pred(mask)"
        else:
            shown_target = target_view
            shown_title = f"{tag}: target"

        resid = (target_view - ((1.0 - pixel_mask) * target_view + pixel_mask * recon_img)).abs().clamp(0, 1)

        # Per-image residual statistics on masked pixels.
        # Note: residual is defined so that unmasked pixels are exactly 0.
        b = int(resid.size(0))
        resid_ann = []
        for i in range(b):
            m = pixel_mask[i] > 0.5
            v = resid[i][m]
            if v.numel() == 0:
                r_min = r_mean = r_max = 0.0
            else:
                r_min = float(v.min().item())
                r_mean = float(v.mean().item())
                r_max = float(v.max().item())
            resid_ann.append(f"min={r_min:.4f}\nmean={r_mean:.4f}\nmax={r_max:.4f}")

        out_path = str(self.vis_dir / f"{tag}_epoch_{epoch:03d}.png")
        save_image_grid(
            [shown_target, pixel_mask, masked_in, recon_img.clamp(0, 1), resid],
            [shown_title, "mask", "masked_in", "recon", "abs_resid"],
            out_path,
            annotations={4: resid_ann},
            panel_vmax={4: 0.05},
        )


    # -------- Core training --------
    def train_one_epoch(self, loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        meter = MetricsAccumulator()

        loss_recon_orig_list = []
        loss_recon_flip_list = []
        loss_recon_total_list = []
        loss_con_list = []
        loss_total_list = []

        losses_total_scalar = []
        losses_con_scalar = []
        vars_mean = []
        vars_min = []

        pbar = tqdm(loader, desc=f"train {epoch}", leave=False)
        lambda_contrast_eff = self._lambda_contrastive_eff(epoch)

        for step, batch in enumerate(pbar):
            x = batch["input"].to(self.device, non_blocking=True)

            plane = batch.get("plane_one_hot", None)
            if plane is None:
                plane = torch.tensor([0.0, 1.0], device=self.device).view(1, 2).repeat(x.size(0), 1)
            else:
                plane = plane.to(self.device, non_blocking=True)

            pixel_mask = sample_masks_anti_mirror(x.size(0), self.cfg.mask, self.device)

            self.opt.zero_grad(set_to_none=True)

            with autocast(
                device_type=self.device.type,
                enabled=(self.cfg.training.amp and self.device.type == "cuda"),
            ):
                recon_raw_orig, recon_raw_flip, z1, z2 = self.model(
                    x,
                    pixel_mask=pixel_mask,
                    plane_one_hot=plane,
                    return_embeddings=True,
                )

                x_flip = flip_lr(x)

                recon_loss_type = getattr(self.cfg.training, "recon_loss", "weighted_bce_logits")
                fg_eps = float(getattr(self.cfg.training, "fg_eps", 0.02))
                fg_weight = float(getattr(self.cfg.training, "fg_weight", 10.0))

                if recon_loss_type == "weighted_bce_logits":
                    if self.cfg.training.enable_masked_loss:
                        loss_recon_orig = masked_bce_logits_weighted(
                            recon_raw_orig, x, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight
                        )
                        loss_recon_flip = masked_bce_logits_weighted(
                            recon_raw_flip, x_flip, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight
                        )
                    else:
                        loss_recon_orig = mixed_bce_logits_weighted(
                            recon_raw_orig, x, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight
                        )
                        loss_recon_flip = mixed_bce_logits_weighted(
                            recon_raw_flip, x_flip, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight
                        )
                else:
                    recon_img_orig = torch.sigmoid(recon_raw_orig.clamp(-10, 10))
                    recon_img_flip = torch.sigmoid(recon_raw_flip.clamp(-10, 10))

                    if self.cfg.training.enable_masked_loss:
                        loss_recon_orig = masked_l1_loss(recon_img_orig, x, pixel_mask)
                        loss_recon_flip = masked_l1_loss(recon_img_flip, x_flip, pixel_mask)
                    else:
                        loss_recon_orig = mixed_l1_loss(recon_img_orig, x, pixel_mask)
                        loss_recon_flip = mixed_l1_loss(recon_img_flip, x_flip, pixel_mask)

                loss_recon_total = loss_recon_orig + loss_recon_flip

                if self.cfg.training.enable_contrastive:
                    loss_con = nt_xent_loss(z1, z2, temperature=self.cfg.training.temperature)
                else:
                    loss_con = torch.zeros((), device=self.device)

                loss_total = (
                    self.cfg.training.lambda_recon * loss_recon_total
                    + lambda_contrast_eff * loss_con
                )

            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

            with torch.no_grad():
                recon_img_orig_metric = torch.sigmoid(recon_raw_orig.clamp(-10, 10))
                recon_img_flip_metric = torch.sigmoid(recon_raw_flip.clamp(-10, 10))

                diff_orig = (x - recon_img_orig_metric).abs()
                diff_flip = (x_flip - recon_img_flip_metric).abs()
                diff_total = (0.5 * (diff_orig + diff_flip)).detach()

                ssim_orig = ssim_index(x.float(), recon_img_orig_metric.float())
                ssim_flip = ssim_index(x_flip.float(), recon_img_flip_metric.float())
                ssim_sum = float((0.5 * (ssim_orig + ssim_flip)).sum().item())

                meter.update(diff_total, pixel_mask, ssim_sum=ssim_sum)

                loss_recon_orig_list.append(float(loss_recon_orig.item()))
                loss_recon_flip_list.append(float(loss_recon_flip.item()))
                loss_recon_total_list.append(float(loss_recon_total.item()))
                loss_con_list.append(float(loss_con.item()))
                loss_total_list.append(float(loss_total.item()))

                losses_total_scalar.append(float(loss_total.item()))
                losses_con_scalar.append(float(loss_con.item()))

                if self.cfg.training.enable_contrastive:
                    mean_var, min_var = compute_embedding_variance([z1.detach(), z2.detach()])
                    vars_mean.append(float(mean_var))
                    vars_min.append(float(min_var))
                else:
                    vars_mean.append(0.0)
                    vars_min.append(0.0)

                if getattr(self.cfg.logging, "log_losses_every_iter", False):
                    pbar.set_postfix({
                        "re_o": f"{loss_recon_orig.item():.4f}",
                        "re_f": f"{loss_recon_flip.item():.4f}",
                        "re_t": f"{loss_recon_total.item():.4f}",
                        "con": f"{loss_con.item():.4f}",
                        "tot": f"{loss_total.item():.4f}",
                    })

        stats = meter.compute()

        decomp = {
            "loss_recon_orig": float(np.mean(loss_recon_orig_list)) if loss_recon_orig_list else 0.0,
            "loss_recon_flip": float(np.mean(loss_recon_flip_list)) if loss_recon_flip_list else 0.0,
            "loss_recon_total": float(np.mean(loss_recon_total_list)) if loss_recon_total_list else 0.0,
            "loss_contrastive": float(np.mean(loss_con_list)) if loss_con_list else 0.0,
            "loss_total": float(np.mean(loss_total_list)) if loss_total_list else 0.0,
        }

        self._append_loss_decomp_csv(epoch, "train", decomp)

        return {
            "loss": float(np.mean(losses_total_scalar)) if losses_total_scalar else 0.0,
            "loss_contrast": float(np.mean(losses_con_scalar)) if losses_con_scalar else 0.0,
            "var_mean": float(np.mean(vars_mean)) if vars_mean else 0.0,
            "var_min": float(np.mean(vars_min)) if vars_min else 0.0,
            "recon_total": float(stats.total_l1),
            "recon_masked": float(stats.masked_l1),
            "recon_unmasked": float(stats.unmasked_l1),
            "ssim": float(stats.ssim),
            **decomp,
        }

    @torch.no_grad()
    def validate(self, loader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        meter = MetricsAccumulator()
        
        lambda_contrast_eff = self._lambda_contrastive_eff(epoch)

        losses_total = []
        loss_recon_orig_list = []
        loss_recon_flip_list = []
        loss_recon_total_list = []
        loss_con_list = []

        vars_mean = []
        vars_min = []

        for batch in tqdm(loader, desc=f"val {epoch}", leave=False):
            x = batch["input"].to(self.device, non_blocking=True)

            plane = batch.get("plane_one_hot", None)
            if plane is None:
                plane = torch.tensor([0.0, 1.0], device=self.device).view(1, 2).repeat(x.size(0), 1)
            else:
                plane = plane.to(self.device, non_blocking=True)

            pixel_mask = sample_masks_anti_mirror(x.size(0), self.cfg.mask, self.device)

            with autocast(
                device_type=self.device.type,
                enabled=(self.cfg.training.amp and self.device.type == "cuda"),
            ):
                recon_raw_orig, recon_raw_flip, z1, z2 = self.model(
                    x,
                    pixel_mask=pixel_mask,
                    plane_one_hot=plane,
                    return_embeddings=self.cfg.training.enable_contrastive,
                )

                x_flip = flip_lr(x)

                recon_loss_type = getattr(self.cfg.training, "recon_loss", "weighted_bce_logits")
                fg_eps = float(getattr(self.cfg.training, "fg_eps", 0.02))
                fg_weight = float(getattr(self.cfg.training, "fg_weight", 10.0))

                if recon_loss_type == "weighted_bce_logits":
                    if self.cfg.training.enable_masked_loss:
                        loss_recon_orig = masked_bce_logits_weighted(
                            recon_raw_orig, x, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight
                        )
                        loss_recon_flip = masked_bce_logits_weighted(
                            recon_raw_flip, x_flip, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight
                        )
                    else:
                        loss_recon_orig = mixed_bce_logits_weighted(
                            recon_raw_orig, x, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight
                        )
                        loss_recon_flip = mixed_bce_logits_weighted(
                            recon_raw_flip, x_flip, pixel_mask, fg_eps=fg_eps, fg_weight=fg_weight
                        )
                else:
                    recon_img_orig = torch.sigmoid(recon_raw_orig.clamp(-10, 10))
                    recon_img_flip = torch.sigmoid(recon_raw_flip.clamp(-10, 10))

                    if self.cfg.training.enable_masked_loss:
                        loss_recon_orig = masked_l1_loss(recon_img_orig, x, pixel_mask)
                        loss_recon_flip = masked_l1_loss(recon_img_flip, x_flip, pixel_mask)
                    else:
                        loss_recon_orig = mixed_l1_loss(recon_img_orig, x, pixel_mask)
                        loss_recon_flip = mixed_l1_loss(recon_img_flip, x_flip, pixel_mask)

                loss_recon_total = loss_recon_orig + loss_recon_flip

                if self.cfg.training.enable_contrastive:
                    loss_con = nt_xent_loss(z1, z2, temperature=self.cfg.training.temperature)
                else:
                    loss_con = torch.zeros((), device=self.device)

                loss_total = (
                    self.cfg.training.lambda_recon * loss_recon_total
                    + lambda_contrast_eff * loss_con
                )

            recon_img_orig_metric = torch.sigmoid(recon_raw_orig.clamp(-10, 10))
            recon_img_flip_metric = torch.sigmoid(recon_raw_flip.clamp(-10, 10))

            diff_orig = (x - recon_img_orig_metric).abs()
            diff_flip = (x_flip - recon_img_flip_metric).abs()
            diff_total = (0.5 * (diff_orig + diff_flip)).detach()

            ssim_orig = ssim_index(x.float(), recon_img_orig_metric.float())
            ssim_flip = ssim_index(x_flip.float(), recon_img_flip_metric.float())
            ssim_sum = float((0.5 * (ssim_orig + ssim_flip)).sum().item())

            meter.update(diff_total, pixel_mask, ssim_sum=ssim_sum)

            losses_total.append(float(loss_total.item()))
            loss_recon_orig_list.append(float(loss_recon_orig.item()))
            loss_recon_flip_list.append(float(loss_recon_flip.item()))
            loss_recon_total_list.append(float(loss_recon_total.item()))
            loss_con_list.append(float(loss_con.item()))

            if self.cfg.training.enable_contrastive:
                mean_var, min_var = compute_embedding_variance([z1.detach(), z2.detach()])
                vars_mean.append(float(mean_var))
                vars_min.append(float(min_var))
            else:
                vars_mean.append(0.0)
                vars_min.append(0.0)

        stats = meter.compute()

        decomp = {
            "loss_recon_orig": float(np.mean(loss_recon_orig_list)) if loss_recon_orig_list else 0.0,
            "loss_recon_flip": float(np.mean(loss_recon_flip_list)) if loss_recon_flip_list else 0.0,
            "loss_recon_total": float(np.mean(loss_recon_total_list)) if loss_recon_total_list else 0.0,
            "loss_contrastive": float(np.mean(loss_con_list)) if loss_con_list else 0.0,
            "loss_total": float(np.mean(losses_total)) if losses_total else 0.0,
        }

        self._append_loss_decomp_csv(epoch, "val", decomp)

        return {
            "loss": float(np.mean(losses_total)) if losses_total else 0.0,
            "loss_contrast": float(np.mean(loss_con_list)) if loss_con_list else 0.0,
            "var_mean": float(np.mean(vars_mean)) if vars_mean else 0.0,
            "var_min": float(np.mean(vars_min)) if vars_min else 0.0,
            "recon_total": float(stats.total_l1),
            "recon_masked": float(stats.masked_l1),
            "recon_unmasked": float(stats.unmasked_l1),
            "ssim": float(stats.ssim),
            **decomp,
        }


    # -------- hooks --------
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

        recon_raw_orig, recon_raw_flip, _, _ = self.model(
            x, pixel_mask=pixel_mask, plane_one_hot=plane, return_embeddings=False
        )

        recon_img_orig = torch.sigmoid(recon_raw_orig.clamp(-10, 10))
        recon_img_flip = torch.sigmoid(recon_raw_flip.clamp(-10, 10))

        # orig grid: target is x
        self._visualize_recon(x, pixel_mask, recon_img_orig, epoch, tag)

        # flip grid: target should be flipped image
        x_flip = flip_lr(x)
        self._visualize_recon(x_flip, pixel_mask, recon_img_flip, epoch, tag + "_flip")


    def maybe_tsne(self, loader, epoch: int):
        if not self.cfg.logging.enable_tsne:
            return
        if (epoch % self.cfg.logging.tsne_every) != 0:
            return
        if self.cfg.logging.tsne_only_if_labeled:
            try:
                b0 = next(iter(loader))
                if not has_labels_in_batch(b0):
                    return
            except Exception:
                return
        out_prefix = str(self.tsne_dir / f"epoch_{epoch:03d}")
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
        class _Wrap(nn.Module):
            def __init__(self, base: SwinUNetDualViewSSLPhase1):
                super().__init__()
                self.base = base

            @torch.no_grad()
            def encoder_embed(self, x: torch.Tensor, mode: str = "bottleneck"):
                B, _, H, W = x.shape
                device = x.device
                M = torch.zeros((B, 1, H, W), device=device, dtype=x.dtype)
                plane = torch.tensor([0.0, 1.0], device=device).view(1, 2).repeat(B, 1)
                b = self.base.encode_bottleneck(x, plane, view=1)
                h = b.mean(dim=(1, 2))
                return None, h

        return _Wrap(self.model).to(self.device)

    def save_checkpoint(self, *, path: Path, epoch: int, best_val: float):
        obj = {
            "epoch": epoch,
            "best_val": float(best_val),
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),  # để resume AMP đúng
            "cfg": asdict(self.cfg),
        }
        torch.save(obj, path)


    def fit(self, train_loader, val_loader):
        best_val = float("inf")

        best_path = self.ckpt_dir / "best.pt"
        latest_path = self.ckpt_dir / "latest.pt"

        for epoch in range(1, self.cfg.training.epochs + 1):
            t0 = time.time()
            tr = self.train_one_epoch(train_loader, epoch)
            va = self.validate(val_loader, epoch)
            dt = time.time() - t0

            self._append_epoch_csv({
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_recon_total": tr["recon_total"],
                "train_recon_masked": tr["recon_masked"],
                "train_recon_unmasked": tr["recon_unmasked"],
                "train_ssim": tr["ssim"],
                "train_loss_contrast": tr["loss_contrast"],
                "train_embed_var_mean": tr["var_mean"],
                "train_embed_var_min": tr["var_min"],
                "val_loss": va["loss"],
                "val_recon_total": va["recon_total"],
                "val_recon_masked": va["recon_masked"],
                "val_recon_unmasked": va["recon_unmasked"],
                "val_ssim": va["ssim"],
            })

            self.maybe_visualize(val_loader, epoch, tag="val")
            self.maybe_tsne(val_loader, epoch)

            # Save latest every epoch (overwrite)
            self.save_checkpoint(path=latest_path, epoch=epoch, best_val=best_val)

            # Save best only when improved (overwrite best.pt)
            if va["loss"] < best_val:
                best_val = va["loss"]
                self.save_checkpoint(path=best_path, epoch=epoch, best_val=best_val)

            plot_training_curves(self.log_csv_path, self.plots_dir)
            plot_loss_decomposition_curves(
                self.loss_decomp_csv_path,
                self.plots_dir,
            )

            print(
                f"[epoch {epoch:03d}] "
                f"train: recon_o={tr['loss_recon_orig']:.4f} recon_f={tr['loss_recon_flip']:.4f} recon_t={tr['loss_recon_total']:.4f} "
                f"con={tr['loss_contrastive']:.4f} total={tr['loss_total']:.4f} | "
                f"val: recon_o={va['loss_recon_orig']:.4f} recon_f={va['loss_recon_flip']:.4f} recon_t={va['loss_recon_total']:.4f} "
                f"con={va['loss_contrastive']:.4f} total={va['loss_total']:.4f} | time={dt:.1f}s"
            )

    def load_checkpoint_weights(self, ckpt_path: Path) -> Dict[str, Any]:
        obj = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(obj["model"], strict=True)
        return obj

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

    trainer = PhaseATrainer(cfg, device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
