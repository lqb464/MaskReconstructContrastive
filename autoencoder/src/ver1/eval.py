from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from .common.cli_utils import run_entrypoint
from .common.metrics import MetricsAccumulator
from .common.recon_compute import compute_recon_losses
from .data.dataset import FolderSubfolderImageDataset, load_label_map_from_csv
from .models.model_utils import flip_lr
from .training.batch_ops import prepare_inputs
from .training.eval_ckpt import dataclass_from_dict, load_checkpoint, resolve_ckpt_path
from .training.metric_compute import update_recon_metrics
from .training.utils import ensure_dir, get_device
from .viz.visualization import run_tsne_visualization, save_image_grid

from .config.experiment import ExperimentConfig
from .models.factory import build_model


@torch.no_grad()
def run_eval(
    *,
    model,
    loader: DataLoader,
    device: torch.device,
    cfg: ExperimentConfig,
) -> Dict[str, float]:
    model.eval()
    meter = MetricsAccumulator()
    losses_total = []
    loss_recon_orig_list = []
    loss_recon_flip_list = []
    loss_recon_total_list = []
    loss_kl_list = []
    lambda_kl = float(getattr(cfg.training, "lambda_kl", 0.0))
    is_vae = str(cfg.model.backbone).lower() == "vae"

    for batch in loader:
        x, plane, pixel_mask = prepare_inputs(batch, device=device, cfg_mask=cfg.mask)
        x_flip = None if cfg.training.single_view else flip_lr(x)

        with autocast(
            device_type=device.type,
            enabled=(cfg.training.amp and device.type == "cuda"),
        ):
            recon_raw_orig, recon_raw_flip, _, _ = model(
                x,
                pixel_mask=pixel_mask,
                plane_one_hot=plane,
            )
            loss_recon_orig, loss_recon_flip, loss_recon_total = compute_recon_losses(
                recon_raw_orig=recon_raw_orig,
                recon_raw_flip=recon_raw_flip,
                x=x,
                x_flip=x_flip,
                pixel_mask=pixel_mask,
                training_cfg=cfg.training,
            )
            loss_kl = getattr(model, "last_kl_loss", None)
            if loss_kl is None or not is_vae:
                loss_kl = torch.zeros((), device=device)
            loss_total = cfg.training.lambda_recon * loss_recon_total + lambda_kl * loss_kl

        update_recon_metrics(
            meter=meter,
            x=x,
            x_flip=x_flip,
            recon_raw_orig=recon_raw_orig,
            recon_raw_flip=recon_raw_flip,
            pixel_mask=pixel_mask,
        )

        losses_total.append(float(loss_total.item()))
        loss_recon_orig_list.append(float(loss_recon_orig.item()))
        loss_recon_flip_list.append(float(loss_recon_flip.item()))
        loss_recon_total_list.append(float(loss_recon_total.item()))
        loss_kl_list.append(float(loss_kl.item()))

    stats = meter.compute()
    decomp = {
        "loss_recon_orig": float(np.mean(loss_recon_orig_list)) if loss_recon_orig_list else 0.0,
        "loss_recon_flip": float(np.mean(loss_recon_flip_list)) if loss_recon_flip_list else 0.0,
        "loss_recon_total": float(np.mean(loss_recon_total_list)) if loss_recon_total_list else 0.0,
        "loss_kl": float(np.mean(loss_kl_list)) if loss_kl_list else 0.0,
        "loss_total": float(np.mean(losses_total)) if losses_total else 0.0,
    }
    return {
        "loss": float(np.mean(losses_total)) if losses_total else 0.0,
        "recon_total": float(stats.total_l1),
        "recon_masked": float(stats.masked_l1),
        "recon_unmasked": float(stats.unmasked_l1),
        "ssim": float(stats.ssim),
        **decomp,
    }


@torch.no_grad()
def visualize_once(*, model, loader: DataLoader, device: torch.device, out_dir: Path, cfg: ExperimentConfig, tag: str = "eval"):
    model.eval()
    batch = next(iter(loader))
    x, plane, pixel_mask = prepare_inputs(batch, device=device, cfg_mask=cfg.mask)
    recon_raw_orig, recon_raw_flip, _, _ = model(x, pixel_mask=pixel_mask, plane_one_hot=plane)
    recon_img_orig = torch.sigmoid(recon_raw_orig.clamp(-10, 10))
    recon_img_flip = torch.sigmoid(recon_raw_flip.clamp(-10, 10)) if recon_raw_flip is not None else None

    def _one(target_view: torch.Tensor, recon_img: torch.Tensor, suffix: str):
        masked_in = target_view * (1.0 - pixel_mask)

        if cfg.training.enable_masked_loss:
            shown_target = (1.0 - pixel_mask) * target_view + pixel_mask * recon_img
            shown_title = f"{tag}{suffix}: target(unmask)+pred(mask)"
        else:
            shown_target = target_view
            shown_title = f"{tag}{suffix}: target"

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

        out_path = str(out_dir / f"{tag}{suffix}.png")
        save_image_grid(
            [shown_target, pixel_mask, masked_in, recon_img.clamp(0, 1), resid],
            [shown_title, "mask", "masked_in", "recon", "abs_resid"],
            out_path,
            annotations={4: resid_ann},
            panel_vmax={4: 0.05},
        )

    _one(x, recon_img_orig, suffix="_orig")

    if not cfg.training.single_view and recon_img_flip is not None:
        x_flip = flip_lr(x)
        _one(x_flip, recon_img_flip, suffix="_flip")


def build_argparser():
    import argparse

    p = argparse.ArgumentParser("Standalone eval from AE/MAE/VAE checkpoint")
    p.add_argument("--ckpt", type=str, required=True, help='Path or "best" or "latest"')
    p.add_argument("--run-dir", type=str, default="", help="Run directory (contains checkpoints/). Optional")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--image-size", type=int, default=0)
    p.add_argument("--plane", type=str, default="auto", choices=["axial", "coronal", "auto"])
    p.add_argument("--label-csv", type=str, default="")
    p.add_argument("--label-path-col", type=str, default="image_path")
    p.add_argument("--label-col", type=str, default="label")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    p.set_defaults(pin_memory=True)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save-vis", action="store_true")
    p.add_argument("--enable-tsne", action="store_true")
    p.add_argument("--tsne-max-items", type=int, default=1000)
    return p


def run(args) -> None:
    device = get_device(args.cpu)
    run_dir = Path(args.run_dir).expanduser() if args.run_dir else None
    ckpt_path = resolve_ckpt_path(args.ckpt, str(run_dir) if run_dir else None)
    obj = load_checkpoint(ckpt_path, device=device)

    raw_cfg = obj.get("cfg", None)
    if not isinstance(raw_cfg, dict):
        raise ValueError("Checkpoint does not contain cfg dict")

    cfg = dataclass_from_dict(ExperimentConfig, raw_cfg)

    if int(args.image_size) > 0:
        cfg.data.image_size = int(args.image_size)
        cfg.mask.image_size = int(args.image_size)
    cfg.data.plane = args.plane

    if run_dir is None:
        run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent

    out_eval_dir = ensure_dir(run_dir / "eval")
    vis_dir = ensure_dir(out_eval_dir / "vis")
    tsne_dir = ensure_dir(out_eval_dir / "tsne")

    label_map = None
    if args.label_csv:
        label_map = load_label_map_from_csv(
            csv_path=args.label_csv,
            root_dir=args.data_root,
            path_col=args.label_path_col,
            label_col=args.label_col,
        )

    ds = FolderSubfolderImageDataset(
        root_dir=args.data_root,
        image_size=cfg.data.image_size,
        plane=cfg.data.plane,
        label_map=label_map,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(obj["model"], strict=True)

    metrics = run_eval(model=model, loader=loader, device=device, cfg=cfg)
    metrics_path = out_eval_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"[eval] metrics saved to {metrics_path}")

    if args.save_vis:
        visualize_once(model=model, loader=loader, device=device, out_dir=vis_dir, cfg=cfg)

    if args.enable_tsne:
        from torch import nn

        class _Wrap(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            @torch.no_grad()
            def encoder_embed(self, x: torch.Tensor, mode: str = "bottleneck"):
                b, _, h, w = x.shape
                m = torch.zeros((b, 1, h, w), device=x.device, dtype=x.dtype)
                plane = torch.tensor([0.0, 1.0], device=x.device).view(1, 2).repeat(b, 1)
                feat = self.base.encode_bottleneck(x, plane, view=1, pixel_mask=m)
                h_vec = feat.mean(dim=(1, 2)) if feat.ndim == 4 else feat.mean(dim=1)
                return None, h_vec

        run_tsne_visualization(
            model=_Wrap(model).to(device),
            loader=loader,
            device=device,
            out_prefix=str(tsne_dir / "eval"),
            max_items=args.tsne_max_items,
            label_val="label",
            data_module=None,
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_entrypoint(build_argparser, run, argv=argv)


if __name__ == "__main__":
    main()
