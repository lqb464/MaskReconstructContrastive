
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torchinfo import summary
from tqdm import tqdm

from .config.experiment import ExperimentConfig
from .common.losses import nt_xent_loss, compute_embedding_variance, vicreg_loss
from .common.metrics import MetricsAccumulator
from .viz.visualization import (
    plot_loss_decomposition_curves,
    plot_training_curves,
    run_tsne_visualization,
    save_image_grid,
)
from .models.model_utils import flip_lr
from .models.swin_unet_dualview_ssl import SwinUNetDualViewSSL
from .models.unet_dualview_ssl import UNetDualViewSSL

from .training.batch_ops import prepare_inputs
from .training.ckpt_io import load_checkpoint_weights, load_checkpoint_weights_filtered, save_checkpoint
from .training.loggers import EpochCSVLogger, LossDecompCSVLogger
from .training.metric_compute import update_recon_metrics
from .common.recon_compute import compute_recon_losses
from .training.utils import ensure_dir, has_labels_in_batch

class Trainer:
    def __init__(self, cfg: ExperimentConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        if device.type == "cuda" and getattr(cfg.data, "image_size", None):
            torch.backends.cudnn.benchmark = True

        if self.cfg.training.single_view:
            if self.cfg.training.enable_contrastive:
                raise Exception("[Error] single_view requires --disable-contrastive")
            if self.cfg.model.enable_saca:
                raise Exception("[Error] single_view does not support SACA; disable SACA or use dual-view")
            if not self.cfg.training.enable_reconstruct:
                raise Exception("[Error] single_view requires --enable-reconstruct")

        out_dir = Path(cfg.logging.out_dir)
        if cfg.logging.run_name:
            out_dir = out_dir / cfg.logging.run_name
        self.out_dir = ensure_dir(out_dir)

        self.ckpt_dir = ensure_dir(
            Path(cfg.logging.ckpt_dir) if cfg.logging.ckpt_dir else (self.out_dir / "checkpoints")
        )
        self.vis_dir = ensure_dir(self.out_dir / "vis")
        self.plots_dir = ensure_dir(self.out_dir / "plots")
        self.tsne_dir = ensure_dir(self.out_dir / "tsne")

        self.log_csv_path = self.out_dir / "epoch_log.csv"
        self.epoch_logger = EpochCSVLogger(self.log_csv_path)

        self.loss_decomp_csv_path = self.out_dir / "loss_decomp.csv"
        self.loss_logger = LossDecompCSVLogger(self.loss_decomp_csv_path)

        backbone = str(getattr(cfg.model, "backbone", "swin")).lower()
        if backbone == "unet":
            self.model = UNetDualViewSSL(
                in_ch=cfg.model.in_ch,
                base_ch=int(getattr(cfg.model, "unet_base_ch", 16)),
                use_gn=bool(getattr(cfg.model, "unet_use_gn", False)),
                use_se=bool(getattr(cfg.model, "unet_use_se", False)),
                enable_reconstruct=cfg.training.enable_reconstruct,
                enable_contrastive=cfg.training.enable_contrastive,
                single_view=cfg.training.single_view,
                enable_saca=bool(getattr(cfg.model, "enable_saca", False)),
                saca_position=str(getattr(cfg.model, "saca_position", "after_stage1")),
                saca_positions=getattr(cfg.model, "saca_positions", None),
                saca_gate_init=float(getattr(cfg.model, "saca_gate_init", 0.0)),
                saca_warmup_epochs=int(getattr(cfg.model, "saca_warmup_epochs", 0)),
            ).to(device)
        else:
            self.model = SwinUNetDualViewSSL(
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
                enable_saca=cfg.model.enable_saca,
                saca_position=cfg.model.saca_position,
                saca_gate_init=cfg.model.saca_gate_init,
                saca_warmup_epochs=cfg.model.saca_warmup_epochs,
                enable_reconstruct=cfg.training.enable_reconstruct,
                enable_contrastive=cfg.training.enable_contrastive,
                contrastive_loss_type=self.cfg.contrast_loss.contrastive_loss_type,
                contrastive_position=self.cfg.contrast_loss.contrastive_position,
                single_view=cfg.training.single_view,
            ).to(device)

        resume_ckpt = getattr(cfg.training, "resume_ckpt", "")
        ckpt_mode = getattr(cfg.training, "ckpt_load_mode", "none")

        if resume_ckpt and ckpt_mode != "none":
            ckpt_path = Path(resume_ckpt)

            if ckpt_mode == "full":
                load_checkpoint_weights(
                    ckpt_path=ckpt_path,
                    device=self.device,
                    model=self.model,
                    strict=True,
                )

            elif ckpt_mode == "encoder_only":
                obj = load_checkpoint_weights_filtered(
                    ckpt_path=ckpt_path,
                    device=self.device,
                    model=self.model,
                    include_prefixes=self.model.encoder_state_dict_prefixes(),
                    exclude_prefixes=("proj_c1", "proj_c2", "proj_c3", "proj"),
                )

                msg = obj.get("_load_msg", None)
                if msg is not None:
                    print("[ckpt] missing_keys:", len(msg["missing_keys"]))
                    print("[ckpt] unexpected_keys:", len(msg["unexpected_keys"]))

                if bool(getattr(cfg.training, "reset_contrastive_proj_head", True)) and bool(cfg.training.enable_contrastive):
                    self.model.reset_contrastive_projection_heads()
                    print("[ckpt] projection heads reset")

        print(self.model)

        try:
            pc = self.model.param_count_breakdown()

            print("[params] total:", pc.get("total", 0))

            print("[params] enc_early_view1:", pc.get("enc_early_view1", 0))
            print("[params] enc_early_view2:", pc.get("enc_early_view2", 0))
            print("[params] saca attention:", pc.get("saca", 0))
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
            B = 1
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
                depth=4,
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

        if getattr(cfg.training, "torch_compile", False) and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self.opt = AdamW(self.model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        self.scaler = GradScaler(enabled=(cfg.training.amp and device.type == "cuda"))

        self.data_module = None

    def _lambda_contrastive_eff(self, epoch: int) -> float:
        """Linear ramp for contrastive weight (unchanged)."""
        base = float(getattr(self.cfg.training, "lambda_contrast", 0.0))
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

    @torch.no_grad()
    def _visualize_recon(
        self,
        target_view: torch.Tensor,
        pixel_mask: torch.Tensor,
        recon_img: torch.Tensor,
        epoch: int,
        tag: str,
    ):
        masked_in = target_view * (1.0 - pixel_mask)

        if self.cfg.training.enable_masked_loss:
            shown_target = (1.0 - pixel_mask) * target_view + pixel_mask * recon_img
            shown_title = f"{tag}: target(unmask)+pred(mask)"
        else:
            shown_target = target_view
            shown_title = f"{tag}: target"

        resid = (target_view - ((1.0 - pixel_mask) * target_view + pixel_mask * recon_img)).abs().clamp(0, 1)

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

    def train_one_epoch(self, loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        meter = MetricsAccumulator()

        loss_recon_orig_sum = torch.zeros((), device=self.device)
        loss_recon_flip_sum = torch.zeros((), device=self.device)
        loss_recon_total_sum = torch.zeros((), device=self.device)
        loss_con_sum = torch.zeros((), device=self.device)
        loss_total_sum = torch.zeros((), device=self.device)
        loss_count = 0
        vars_mean_sum = 0.0
        vars_min_sum = 0.0

        pbar = tqdm(loader, desc=f"train {epoch}", leave=False)
        lambda_contrast_eff = self._lambda_contrastive_eff(epoch)

        for batch in pbar:
            x, plane, pixel_mask = prepare_inputs(batch, device=self.device, cfg_mask=self.cfg.mask)

            self.opt.zero_grad(set_to_none=True)

            with autocast(
                device_type=self.device.type,
                enabled=(self.cfg.training.amp and self.device.type == "cuda"),
            ):
                recon_raw_orig, recon_raw_flip, z1, z2 = self.model(
                    x,
                    pixel_mask=pixel_mask,
                    plane_one_hot=plane,
                )
                x_flip = None
                if self.cfg.training.enable_reconstruct and (not self.cfg.training.single_view):
                    x_flip = flip_lr(x)

                if self.cfg.training.enable_reconstruct:
                    loss_recon_orig, loss_recon_flip, loss_recon_total = compute_recon_losses(
                        recon_raw_orig=recon_raw_orig,
                        recon_raw_flip=recon_raw_flip,
                        x=x,
                        x_flip=x_flip,
                        pixel_mask=pixel_mask,
                        training_cfg=self.cfg.training,
                    )
                else:
                    loss_recon_orig = torch.zeros((), device=self.device)
                    loss_recon_flip = torch.zeros((), device=self.device)
                    loss_recon_total = torch.zeros((), device=self.device)

                if self.cfg.training.enable_contrastive:
                    if self.cfg.contrast_loss.contrastive_loss_type == "infonce":
                        loss_con = nt_xent_loss(
                            z1=z1,
                            z2=z2,
                            temperature=self.cfg.training.temperature
                        )
                    elif self.cfg.contrast_loss.contrastive_loss_type == "vicreg":
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
                    loss_con = torch.zeros((), device=self.device)

                if self.cfg.training.enable_contrastive and self.cfg.training.enable_reconstruct:
                    loss_total = self.cfg.training.lambda_recon * loss_recon_total + lambda_contrast_eff * loss_con

                elif not self.cfg.training.enable_contrastive:
                    loss_total = self.cfg.training.lambda_recon * loss_recon_total

                elif not self.cfg.training.enable_reconstruct:
                    loss_total = lambda_contrast_eff * loss_con

            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

            with torch.no_grad():
                if self.cfg.training.enable_reconstruct:
                    update_recon_metrics(
                        meter=meter,
                        x=x,
                        x_flip=x_flip,
                        recon_raw_orig=recon_raw_orig,
                        recon_raw_flip=recon_raw_flip,
                        pixel_mask=pixel_mask,
                    )

                loss_recon_orig_sum += loss_recon_orig.detach()
                loss_recon_flip_sum += loss_recon_flip.detach()
                loss_recon_total_sum += loss_recon_total.detach()
                loss_con_sum += loss_con.detach()
                loss_total_sum += loss_total.detach()
                loss_count += 1

                if self.cfg.training.enable_contrastive:
                    mean_var, min_var = compute_embedding_variance([z1.detach(), z2.detach()])
                    vars_mean_sum += float(mean_var)
                    vars_min_sum += float(min_var)
                else:
                    vars_mean_sum += 0.0
                    vars_min_sum += 0.0

                if getattr(self.cfg.logging, "log_losses_every_iter", False):
                    pbar.set_postfix({
                        "re_o": f"{loss_recon_orig.item():.4f}",
                        "re_f": f"{loss_recon_flip.item():.4f}",
                        "re_t": f"{loss_recon_total.item():.4f}",
                        "con": f"{loss_con.item():.4f}",
                        "tot": f"{loss_total.item():.4f}",
                    })

        stats = meter.compute()

        if loss_count:
            loss_recon_orig_mean = (loss_recon_orig_sum / loss_count).item()
            loss_recon_flip_mean = (loss_recon_flip_sum / loss_count).item()
            loss_recon_total_mean = (loss_recon_total_sum / loss_count).item()
            loss_con_mean = (loss_con_sum / loss_count).item()
            loss_total_mean = (loss_total_sum / loss_count).item()
        else:
            loss_recon_orig_mean = 0.0
            loss_recon_flip_mean = 0.0
            loss_recon_total_mean = 0.0
            loss_con_mean = 0.0
            loss_total_mean = 0.0

        decomp = {
            "loss_recon_orig": loss_recon_orig_mean,
            "loss_recon_flip": loss_recon_flip_mean,
            "loss_recon_total": loss_recon_total_mean,
            "loss_contrastive": loss_con_mean,
            "loss_total": loss_total_mean,
        }

        self.loss_logger.append(epoch, "train", decomp)

        return {
            "loss": loss_total_mean,
            "loss_contrast": loss_con_mean,
            "var_mean": (vars_mean_sum / loss_count) if loss_count else 0.0,
            "var_min": (vars_min_sum / loss_count) if loss_count else 0.0,
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

        loss_total_sum = torch.zeros((), device=self.device)
        loss_recon_orig_sum = torch.zeros((), device=self.device)
        loss_recon_flip_sum = torch.zeros((), device=self.device)
        loss_recon_total_sum = torch.zeros((), device=self.device)
        loss_con_sum = torch.zeros((), device=self.device)
        loss_count = 0
        vars_mean_sum = 0.0
        vars_min_sum = 0.0

        for batch in tqdm(loader, desc=f"val {epoch}", leave=False):
            x, plane, pixel_mask = prepare_inputs(batch, device=self.device, cfg_mask=self.cfg.mask)

            with autocast(
                device_type=self.device.type,
                enabled=(self.cfg.training.amp and self.device.type == "cuda"),
            ):
                recon_raw_orig, recon_raw_flip, z1, z2 = self.model(
                    x,
                    pixel_mask=pixel_mask,
                    plane_one_hot=plane,
                )
                x_flip = None
                if self.cfg.training.enable_reconstruct and (not self.cfg.training.single_view):
                    x_flip = flip_lr(x)

                if self.cfg.training.enable_reconstruct:
                    loss_recon_orig, loss_recon_flip, loss_recon_total = compute_recon_losses(
                        recon_raw_orig=recon_raw_orig,
                        recon_raw_flip=recon_raw_flip,
                        x=x,
                        x_flip=x_flip,
                        pixel_mask=pixel_mask,
                        training_cfg=self.cfg.training,
                    )
                else:
                    loss_recon_orig = torch.zeros((), device=self.device)
                    loss_recon_flip = torch.zeros((), device=self.device)
                    loss_recon_total = torch.zeros((), device=self.device)

                if self.cfg.training.enable_contrastive:
                    if self.cfg.contrast_loss.contrastive_loss_type == "infonce":
                        loss_con = nt_xent_loss(
                            z1=z1,
                            z2=z2,
                            temperature=self.cfg.training.temperature
                        )
                    elif self.cfg.contrast_loss.contrastive_loss_type == "vicreg":
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
                    loss_con = torch.zeros((), device=self.device)

                if self.cfg.training.enable_contrastive and self.cfg.training.enable_reconstruct:
                    loss_total = self.cfg.training.lambda_recon * loss_recon_total + lambda_contrast_eff * loss_con

                elif not self.cfg.training.enable_contrastive:
                    loss_total = self.cfg.training.lambda_recon * loss_recon_total

                elif not self.cfg.training.enable_reconstruct:
                    loss_total = lambda_contrast_eff * loss_con

            if self.cfg.training.enable_reconstruct:
                update_recon_metrics(
                    meter=meter,
                    x=x,
                    x_flip=x_flip,
                    recon_raw_orig=recon_raw_orig,
                    recon_raw_flip=recon_raw_flip,
                    pixel_mask=pixel_mask,
                )

            loss_total_sum += loss_total.detach()
            loss_recon_orig_sum += loss_recon_orig.detach()
            loss_recon_flip_sum += loss_recon_flip.detach()
            loss_recon_total_sum += loss_recon_total.detach()
            loss_con_sum += loss_con.detach()
            loss_count += 1

            if self.cfg.training.enable_contrastive:
                mean_var, min_var = compute_embedding_variance([z1.detach(), z2.detach()])
                vars_mean_sum += float(mean_var)
                vars_min_sum += float(min_var)
            else:
                vars_mean_sum += 0.0
                vars_min_sum += 0.0

        stats = meter.compute()

        if loss_count:
            loss_recon_orig_mean = (loss_recon_orig_sum / loss_count).item()
            loss_recon_flip_mean = (loss_recon_flip_sum / loss_count).item()
            loss_recon_total_mean = (loss_recon_total_sum / loss_count).item()
            loss_con_mean = (loss_con_sum / loss_count).item()
            loss_total_mean = (loss_total_sum / loss_count).item()
        else:
            loss_recon_orig_mean = 0.0
            loss_recon_flip_mean = 0.0
            loss_recon_total_mean = 0.0
            loss_con_mean = 0.0
            loss_total_mean = 0.0

        decomp = {
            "loss_recon_orig": loss_recon_orig_mean,
            "loss_recon_flip": loss_recon_flip_mean,
            "loss_recon_total": loss_recon_total_mean,
            "loss_contrastive": loss_con_mean,
            "loss_total": loss_total_mean,
        }

        self.loss_logger.append(epoch, "val", decomp)

        return {
            "loss": loss_total_mean,
            "loss_contrast": loss_con_mean,
            "var_mean": (vars_mean_sum / loss_count) if loss_count else 0.0,
            "var_min": (vars_min_sum / loss_count) if loss_count else 0.0,
            "recon_total": float(stats.total_l1),
            "recon_masked": float(stats.masked_l1),
            "recon_unmasked": float(stats.unmasked_l1),
            "ssim": float(stats.ssim),
            **decomp,
        }

    def maybe_visualize(self, loader, epoch: int, tag: str):
        if not self.cfg.training.enable_reconstruct:
            return

        if (epoch % self.cfg.logging.vis_every) != 0:
            return
        self.model.eval()
        batch = next(iter(loader))
        x, plane, pixel_mask = prepare_inputs(batch, device=self.device, cfg_mask=self.cfg.mask)

        recon_raw_orig, recon_raw_flip, _, _ = self.model(
            x, pixel_mask=pixel_mask, plane_one_hot=plane,
        )

        recon_img_orig = torch.sigmoid(recon_raw_orig.clamp(-10, 10))

        self._visualize_recon(x, pixel_mask, recon_img_orig, epoch, tag)

        if not self.cfg.training.single_view:
            recon_img_flip = torch.sigmoid(recon_raw_flip.clamp(-10, 10))
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
            def __init__(self, base: SwinUNetDualViewSSL):
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
        save_checkpoint(
            path=path,
            epoch=epoch,
            best_val=best_val,
            model=self.model,
            optimizer=self.opt,
            scaler=self.scaler,
            cfg=self.cfg,
        )

    def fit(self, train_loader, val_loader):
        best_val = float("inf")

        best_path = self.ckpt_dir / "best.pt"
        latest_path = self.ckpt_dir / "latest.pt"
        save_latest_every = int(getattr(self.cfg.logging, "save_latest_every", 1))
        save_best_after_epoch = int(getattr(self.cfg.logging, "save_best_after_epoch", 0))
        save_best_every = int(getattr(self.cfg.logging, "save_best_every", 1))
        if save_latest_every <= 0:
            save_latest_every = 1
        if save_best_every <= 0:
            save_best_every = 1

        for epoch in range(1, self.cfg.training.epochs + 1):

            freeze_n = int(getattr(self.cfg.training, "freeze_encoder_epochs", 0))
            self.model.set_encoder_trainable(trainable=not (epoch <= freeze_n))

            t0 = time.time()
            tr = self.train_one_epoch(train_loader, epoch)
            va = self.validate(val_loader, epoch)
            dt = time.time() - t0

            self.epoch_logger.append({
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

            if epoch >= save_best_after_epoch and (epoch % save_best_every == 0):
                if va["loss"] < best_val:
                    best_val = va["loss"]
                    self.save_checkpoint(path=best_path, epoch=epoch, best_val=best_val)

            if (epoch % save_latest_every == 0) or (epoch == self.cfg.training.epochs):
                self.save_checkpoint(path=latest_path, epoch=epoch, best_val=best_val)

            print(
                f"[epoch {epoch:03d}] \n"
                f"Train: recon_o={tr['loss_recon_orig']:.4f} recon_f={tr['loss_recon_flip']:.4f} recon_t={tr['loss_recon_total']:.4f} "
                f"con={tr['loss_contrastive']:.4f} total={tr['loss_total']:.4f} \n"
                f"Val: recon_o={va['loss_recon_orig']:.4f} recon_f={va['loss_recon_flip']:.4f} recon_t={va['loss_recon_total']:.4f} "
                f"con={va['loss_contrastive']:.4f} total={va['loss_total']:.4f} | time={dt:.1f}s"
            )

        plot_training_curves(self.log_csv_path, self.plots_dir)
        plot_loss_decomposition_curves(
            self.loss_decomp_csv_path,
            self.plots_dir,
        )

    def load_checkpoint_weights(self, ckpt_path: Path) -> Dict[str, Any]:
        return load_checkpoint_weights(ckpt_path=ckpt_path, device=self.device, model=self.model, strict=True)
