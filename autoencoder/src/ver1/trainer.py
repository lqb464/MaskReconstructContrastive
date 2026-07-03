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

from .common.metrics import MetricsAccumulator
from .common.recon_compute import compute_recon_losses
from .models.model_utils import flip_lr
from .training.batch_ops import prepare_inputs
from .training.ckpt_io import load_checkpoint_weights, load_checkpoint_weights_filtered, save_checkpoint
from .training.loggers import EpochCSVLogger, LossDecompCSVLogger
from .training.metric_compute import update_recon_metrics
from .training.utils import ensure_dir, has_labels_in_batch
from .viz.visualization import (
    plot_loss_decomposition_curves,
    plot_training_curves,
    run_tsne_visualization,
    save_image_grid,
)

from .config.experiment import ExperimentConfig
from .models.factory import build_model


class Trainer:
    def __init__(self, cfg: ExperimentConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        if device.type == "cuda" and getattr(cfg.data, "image_size", None):
            torch.backends.cudnn.benchmark = True

        if self.cfg.training.single_view and not self.cfg.training.enable_reconstruct:
            raise Exception("[Error] single_view requires --enable-reconstruct")

        if self.cfg.training.enable_contrastive:
            raise Exception("[Error] AutoEncoder backbones currently support reconstruction-only training.")

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

        self.model = build_model(cfg).to(device)
        self._is_vae = str(cfg.model.backbone).lower() == "vae"

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

        print(self.model)

        try:
            pc = self.model.param_count_breakdown()
            print("[params] total:", pc.get("total", 0))
            print("[params] encoder:", pc.get("enc_early_view1", 0))
            print("[params] decoder:", pc.get("decoder_shared_up2", 0))
        except Exception as e:
            print("[params] unable to compute breakdown:", repr(e))

        try:
            b = 1
            h = cfg.data.image_size
            w = cfg.data.image_size
            in_ch = cfg.model.in_ch
            dummy_x = torch.zeros(b, in_ch, h, w, device=device)
            dummy_pixel_mask = torch.zeros(b, 1, h, w, device=device)
            dummy_plane_one_hot = torch.zeros(b, 2, device=device)
            print("\n[torchinfo] Model architecture summary\n")
            summary(
                self.model,
                input_data=(dummy_x, dummy_pixel_mask, dummy_plane_one_hot),
                depth=4,
                col_names=("input_size", "output_size", "num_params", "trainable"),
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

    def _kl_loss(self) -> torch.Tensor:
        if not self._is_vae:
            return torch.zeros((), device=self.device)
        kl = getattr(self.model, "last_kl_loss", None)
        if kl is None:
            return torch.zeros((), device=self.device)
        return kl

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
        loss_kl_sum = torch.zeros((), device=self.device)
        loss_total_sum = torch.zeros((), device=self.device)
        loss_count = 0

        pbar = tqdm(loader, desc=f"train {epoch}", leave=False)
        lambda_kl = float(getattr(self.cfg.training, "lambda_kl", 0.0))

        for batch in pbar:
            x, plane, pixel_mask = prepare_inputs(batch, device=self.device, cfg_mask=self.cfg.mask)
            self.opt.zero_grad(set_to_none=True)

            with autocast(
                device_type=self.device.type,
                enabled=(self.cfg.training.amp and self.device.type == "cuda"),
            ):
                recon_raw_orig, recon_raw_flip, _, _ = self.model(
                    x,
                    pixel_mask=pixel_mask,
                    plane_one_hot=plane,
                )
                x_flip = None
                if not self.cfg.training.single_view:
                    x_flip = flip_lr(x)

                loss_recon_orig, loss_recon_flip, loss_recon_total = compute_recon_losses(
                    recon_raw_orig=recon_raw_orig,
                    recon_raw_flip=recon_raw_flip,
                    x=x,
                    x_flip=x_flip,
                    pixel_mask=pixel_mask,
                    training_cfg=self.cfg.training,
                )
                loss_kl = self._kl_loss()
                loss_total = (
                    self.cfg.training.lambda_recon * loss_recon_total
                    + lambda_kl * loss_kl
                )

            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

            with torch.no_grad():
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
                loss_kl_sum += loss_kl.detach()
                loss_total_sum += loss_total.detach()
                loss_count += 1

        stats = meter.compute()

        if loss_count:
            loss_recon_orig_mean = (loss_recon_orig_sum / loss_count).item()
            loss_recon_flip_mean = (loss_recon_flip_sum / loss_count).item()
            loss_recon_total_mean = (loss_recon_total_sum / loss_count).item()
            loss_kl_mean = (loss_kl_sum / loss_count).item()
            loss_total_mean = (loss_total_sum / loss_count).item()
        else:
            loss_recon_orig_mean = 0.0
            loss_recon_flip_mean = 0.0
            loss_recon_total_mean = 0.0
            loss_kl_mean = 0.0
            loss_total_mean = 0.0

        decomp = {
            "loss_recon_orig": loss_recon_orig_mean,
            "loss_recon_flip": loss_recon_flip_mean,
            "loss_recon_total": loss_recon_total_mean,
            "loss_kl": loss_kl_mean,
            "loss_contrastive": 0.0,
            "loss_total": loss_total_mean,
        }
        self.loss_logger.append(epoch, "train", decomp)

        return {
            "loss": loss_total_mean,
            "loss_contrast": 0.0,
            "var_mean": 0.0,
            "var_min": 0.0,
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
        lambda_kl = float(getattr(self.cfg.training, "lambda_kl", 0.0))

        loss_total_sum = torch.zeros((), device=self.device)
        loss_recon_orig_sum = torch.zeros((), device=self.device)
        loss_recon_flip_sum = torch.zeros((), device=self.device)
        loss_recon_total_sum = torch.zeros((), device=self.device)
        loss_kl_sum = torch.zeros((), device=self.device)
        loss_count = 0

        for batch in tqdm(loader, desc=f"val {epoch}", leave=False):
            x, plane, pixel_mask = prepare_inputs(batch, device=self.device, cfg_mask=self.cfg.mask)

            with autocast(
                device_type=self.device.type,
                enabled=(self.cfg.training.amp and self.device.type == "cuda"),
            ):
                recon_raw_orig, recon_raw_flip, _, _ = self.model(
                    x,
                    pixel_mask=pixel_mask,
                    plane_one_hot=plane,
                )
                x_flip = None
                if not self.cfg.training.single_view:
                    x_flip = flip_lr(x)

                loss_recon_orig, loss_recon_flip, loss_recon_total = compute_recon_losses(
                    recon_raw_orig=recon_raw_orig,
                    recon_raw_flip=recon_raw_flip,
                    x=x,
                    x_flip=x_flip,
                    pixel_mask=pixel_mask,
                    training_cfg=self.cfg.training,
                )
                loss_kl = self._kl_loss()
                loss_total = self.cfg.training.lambda_recon * loss_recon_total + lambda_kl * loss_kl

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
            loss_kl_sum += loss_kl.detach()
            loss_count += 1

        stats = meter.compute()

        if loss_count:
            loss_recon_orig_mean = (loss_recon_orig_sum / loss_count).item()
            loss_recon_flip_mean = (loss_recon_flip_sum / loss_count).item()
            loss_recon_total_mean = (loss_recon_total_sum / loss_count).item()
            loss_kl_mean = (loss_kl_sum / loss_count).item()
            loss_total_mean = (loss_total_sum / loss_count).item()
        else:
            loss_recon_orig_mean = 0.0
            loss_recon_flip_mean = 0.0
            loss_recon_total_mean = 0.0
            loss_kl_mean = 0.0
            loss_total_mean = 0.0

        decomp = {
            "loss_recon_orig": loss_recon_orig_mean,
            "loss_recon_flip": loss_recon_flip_mean,
            "loss_recon_total": loss_recon_total_mean,
            "loss_kl": loss_kl_mean,
            "loss_contrastive": 0.0,
            "loss_total": loss_total_mean,
        }
        self.loss_logger.append(epoch, "val", decomp)

        return {
            "loss": loss_total_mean,
            "loss_contrast": 0.0,
            "var_mean": 0.0,
            "var_min": 0.0,
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

        class _Wrap(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            @torch.no_grad()
            def encoder_embed(self, x: torch.Tensor, mode: str = "bottleneck"):
                b, _, h, w = x.shape
                device = x.device
                m = torch.zeros((b, 1, h, w), device=device, dtype=x.dtype)
                plane = torch.tensor([0.0, 1.0], device=device).view(1, 2).repeat(b, 1)
                feat = self.base.encode_bottleneck(x, plane, view=1, pixel_mask=m)
                h_vec = feat.mean(dim=(1, 2)) if feat.ndim == 4 else feat.mean(dim=1)
                return None, h_vec

        run_tsne_visualization(
            model=_Wrap(self.model).to(self.device),
            loader=loader,
            device=self.device,
            out_prefix=out_prefix,
            max_items=self.cfg.logging.tsne_max_items,
            label_val="label",
            data_module=self.data_module,
        )

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

            kl_tr = tr.get("loss_kl", 0.0)
            kl_va = va.get("loss_kl", 0.0)
            print(
                f"[epoch {epoch:03d}] \n"
                f"Train: recon_o={tr['loss_recon_orig']:.4f} recon_f={tr['loss_recon_flip']:.4f} "
                f"recon_t={tr['loss_recon_total']:.4f} kl={kl_tr:.4f} total={tr['loss_total']:.4f} \n"
                f"Val: recon_o={va['loss_recon_orig']:.4f} recon_f={va['loss_recon_flip']:.4f} "
                f"recon_t={va['loss_recon_total']:.4f} kl={kl_va:.4f} total={va['loss_total']:.4f} | time={dt:.1f}s"
            )

        plot_training_curves(self.log_csv_path, self.plots_dir)
        plot_loss_decomposition_curves(self.loss_decomp_csv_path, self.plots_dir)

    def load_checkpoint_weights(self, ckpt_path: Path) -> Dict[str, Any]:
        return load_checkpoint_weights(ckpt_path=ckpt_path, device=self.device, model=self.model, strict=True)
