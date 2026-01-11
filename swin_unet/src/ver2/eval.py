# =============================================
# File: eval.py
# Standalone evaluation script
# - Load checkpoint (best/latest/path)
# - Run inference on another folder dataset
# - Compute loss + metrics
# - Save visualization and tSNE
# =============================================
from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, get_origin, get_args, get_type_hints, Union


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from .data.augmentation import sample_masks_anti_mirror
from .config.experiment import ExperimentConfig
from .data.dataset import FolderSubfolderImageDataset, load_label_map_from_csv
from .common.losses import masked_l1_loss, mixed_l1_loss, nt_xent_loss, compute_embedding_variance, ssim_index
from .common.metrics import MetricsAccumulator
from .model.swin_unet_dualview import SwinUNetDualViewSSLPhase1, flip_lr
from .viz.visualization import save_image_grid, run_tsne_visualization

from .training.utils import get_device, ensure_dir
from .common.losses import masked_bce_logits_weighted, mixed_bce_logits_weighted


def dataclass_from_dict(dc_type, raw: dict):
    """
    Rebuild nested dataclass from dict, resolving forward refs via get_type_hints().
    """
    if not is_dataclass(dc_type):
        raise TypeError(f"{dc_type} is not a dataclass")

    # IMPORTANT: resolve real field types (handles postponed annotations / ForwardRef)
    type_hints = get_type_hints(dc_type)

    kwargs = {}
    for f in fields(dc_type):
        name = f.name
        if name not in raw:
            continue

        val = raw[name]
        ftype = type_hints.get(name, f.type)

        # Case 1: nested dataclass
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[name] = dataclass_from_dict(ftype, val)
            continue

        # Case 2: Optional[Dataclass] or Union[Dataclass, None]
        origin = get_origin(ftype)
        args = get_args(ftype)
        if origin is Union and isinstance(val, dict):
            dc_candidates = [a for a in args if is_dataclass(a)]
            if dc_candidates:
                kwargs[name] = dataclass_from_dict(dc_candidates[0], val)
                continue

        kwargs[name] = val

    return dc_type(**kwargs)


def resolve_ckpt_path(ckpt: str, ckpt_dir: Optional[str]) -> Path:
    """
    ckpt: "best" | "latest" | path
    """
    s = (ckpt or "").strip()
    if s.lower() in {"best", "latest"}:
        if not ckpt_dir:
            raise ValueError("ckpt_dir is empty in checkpoint cfg, cannot resolve 'best' or 'latest'")
        p = Path(ckpt_dir) / "checkpoints" / f"{s.lower()}.pt"
        # In trainer, ckpt_dir defaults to out_dir/checkpoints, but cfg.logging.ckpt_dir can override.
        # We try both common layouts:
        if p.exists():
            return p
        p2 = Path(ckpt_dir) / f"{s.lower()}.pt"
        if p2.exists():
            return p2
        raise FileNotFoundError(f"Cannot resolve checkpoint '{s}' under: {ckpt_dir}")
    return Path(s).expanduser()


def load_checkpoint(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    obj = torch.load(ckpt_path, map_location=device)
    if not isinstance(obj, dict) or "model" not in obj:
        raise ValueError(f"Invalid checkpoint format: {ckpt_path}")
    return obj


# -------------------------
# Visualization (same style as trainer)
# -------------------------
@torch.no_grad()
def visualize_once(
    *,
    model: SwinUNetDualViewSSLPhase1,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    cfg: ExperimentConfig,
    tag: str = "eval",
):
    model.eval()
    batch = next(iter(loader))
    x = batch["input"].to(device, non_blocking=True)

    plane = batch.get("plane_one_hot", None)
    if plane is None:
        plane = torch.tensor([0.0, 1.0], device=device).view(1, 2).repeat(x.size(0), 1)
    else:
        plane = plane.to(device, non_blocking=True)

    pixel_mask = sample_masks_anti_mirror(x.size(0), cfg.mask, device)

    recon_raw_orig, recon_raw_flip, _, _ = model(
        x, pixel_mask=pixel_mask, plane_one_hot=plane, return_embeddings=False
    )

    recon_img_orig = torch.sigmoid(recon_raw_orig.clamp(-10, 10))
    recon_img_flip = torch.sigmoid(recon_raw_flip.clamp(-10, 10))

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
    x_flip = flip_lr(x)
    _one(x_flip, recon_img_flip, suffix="_flip")


def tsne_wrapper_model(base: SwinUNetDualViewSSLPhase1, device: torch.device) -> nn.Module:
    class _Wrap(nn.Module):
        def __init__(self, m: SwinUNetDualViewSSLPhase1):
            super().__init__()
            self.base = m

        @torch.no_grad()
        def encoder_embed(self, x: torch.Tensor, mode: str = "bottleneck"):
            B, _, H, W = x.shape
            plane = torch.tensor([0.0, 1.0], device=x.device).view(1, 2).repeat(B, 1)
            b = self.base.encode_bottleneck(x, plane, view=1)
            h = b.mean(dim=(1, 2))
            return None, h

    return _Wrap(base).to(device)


# -------------------------
# Eval core (validate-like)
# -------------------------
@torch.no_grad()
def run_eval(
    *,
    model: SwinUNetDualViewSSLPhase1,
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
    loss_con_list = []
    vars_mean = []
    vars_min = []

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)

        plane = batch.get("plane_one_hot", None)
        if plane is None:
            plane = torch.tensor([0.0, 1.0], device=device).view(1, 2).repeat(x.size(0), 1)
        else:
            plane = plane.to(device, non_blocking=True)

        pixel_mask = sample_masks_anti_mirror(x.size(0), cfg.mask, device)

        with autocast(
            device_type=device.type,
            enabled=(cfg.training.amp and device.type == "cuda"),
        ):
            recon_raw_orig, recon_raw_flip, z1, z2 = model(
                x,
                pixel_mask=pixel_mask,
                plane_one_hot=plane,
                return_embeddings=cfg.training.enable_contrastive,
            )

            x_flip = flip_lr(x)

            recon_loss_type = getattr(cfg.training, "recon_loss", "weighted_bce_logits")
            fg_eps = float(getattr(cfg.training, "fg_eps", 0.02))
            fg_weight = float(getattr(cfg.training, "fg_weight", 10.0))

            if recon_loss_type == "weighted_bce_logits":
                if cfg.training.enable_masked_loss:
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
                if cfg.training.enable_masked_loss:
                    loss_recon_orig = masked_l1_loss(recon_img_orig, x, pixel_mask)
                    loss_recon_flip = masked_l1_loss(recon_img_flip, x_flip, pixel_mask)
                else:
                    loss_recon_orig = mixed_l1_loss(recon_img_orig, x, pixel_mask)
                    loss_recon_flip = mixed_l1_loss(recon_img_flip, x_flip, pixel_mask)

            loss_recon_total = loss_recon_orig + loss_recon_flip

            if cfg.training.enable_contrastive:
                loss_con = nt_xent_loss(z1, z2, temperature=cfg.training.temperature)
            else:
                loss_con = torch.zeros((), device=device)

            loss_total = (
                cfg.training.lambda_recon * loss_recon_total
                + cfg.training.lambda_contrast * loss_con
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

        if cfg.training.enable_contrastive:
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


# -------------------------
# CLI
# -------------------------
def build_argparser():
    import argparse

    p = argparse.ArgumentParser("Standalone eval from checkpoint")
    p.add_argument("--ckpt", type=str, required=True, help='Path or "best" or "latest"')
    p.add_argument("--run-dir", type=str, default="", help="Run directory (contains checkpoints/, vis/, tsne/). Optional")

    p.add_argument("--data-root", type=str, required=True, help="Eval folder root (subfolders of .png)")
    p.add_argument("--image-size", type=int, default=0, help="Override image size (0 uses checkpoint cfg)")
    p.add_argument("--plane", type=str, default="auto", choices=["axial", "coronal", "auto"])

    p.add_argument("--label-csv", type=str, default="", help="Optional label csv for eval dataset")
    p.add_argument("--label-path-col", type=str, default="image_path")
    p.add_argument("--label-col", type=str, default="label")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    p.set_defaults(pin_memory=True)

    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save-vis", action="store_true", help="Save recon visualizations once")
    p.add_argument("--enable-tsne", action="store_true", help="Run tSNE once")
    p.add_argument("--tsne-max-items", type=int, default=1000)

    return p


def main():
    args = build_argparser().parse_args()
    device = get_device(args.cpu)

    run_dir = Path(args.run_dir).expanduser() if args.run_dir else None

    # if ckpt is best/latest and run_dir provided, resolve from there
    if args.ckpt.lower() in {"best", "latest"}:
        if run_dir is None:
            raise ValueError("When ckpt is best/latest, you must provide --run-dir")
        ckpt_path = (run_dir / "checkpoints" / f"{args.ckpt.lower()}.pt").expanduser()
    else:
        ckpt_path = Path(args.ckpt).expanduser()

    obj = load_checkpoint(ckpt_path, device=device)

    # rebuild cfg from checkpoint
    raw_cfg = obj.get("cfg", None)
    if not isinstance(raw_cfg, dict):
        raise ValueError("Checkpoint does not contain cfg dict")

    cfg = dataclass_from_dict(ExperimentConfig, raw_cfg)

    # optional overrides
    if int(args.image_size) > 0:
        cfg.data.image_size = int(args.image_size)
        cfg.mask.image_size = int(args.image_size)

    cfg.data.plane = args.plane

    # output folders
    if run_dir is None:
        # default to checkpoint parent run folder guess: .../checkpoints/best.pt -> parent parent
        run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent

    out_eval_dir = ensure_dir(run_dir / "eval")
    vis_dir = ensure_dir(out_eval_dir / "vis")
    tsne_dir = ensure_dir(out_eval_dir / "tsne")

    # dataset and loader
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
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=False,
    )

    # build model and load weights
    model = SwinUNetDualViewSSLPhase1(
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
    ).to(device)

    model.load_state_dict(obj["model"], strict=True)

    # run eval
    metrics = run_eval(model=model, loader=loader, device=device, cfg=cfg)

    # save metrics
    metrics_out = {
        "ckpt_path": str(ckpt_path),
        "ckpt_epoch": int(obj.get("epoch", 0)),
        "best_val_at_save": float(obj.get("best_val", float("inf"))),
        "num_items": int(len(ds)),
        "metrics": metrics,
    }
    (out_eval_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    print("Eval done.")
    print(json.dumps(metrics_out, indent=2))

    # vis once
    if args.save_vis:
        visualize_once(
            model=model,
            loader=loader,
            device=device,
            out_dir=vis_dir,
            cfg=cfg,
            tag="eval",
        )

    # tsne once
    if args.enable_tsne:
        wrap = tsne_wrapper_model(model, device=device)
        out_prefix = str(tsne_dir / "eval")
        run_tsne_visualization(
            model=wrap,
            loader=loader,
            device=device,
            out_prefix=out_prefix,
            max_items=int(args.tsne_max_items),
            label_val="label",
            data_module=None,
        )


if __name__ == "__main__":
    main()
