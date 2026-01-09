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
from torch.utils.data import DataLoader


from data import FolderDataset, get_eval_transforms
from model import SwinUnetDualViewSSL, flip_lr
from losses import (
    masked_l1_loss,
    mixed_l1_loss,
    ssim_index,
    nt_xent_loss,
    variance_regularizer,
)
from metrics import MetricsAccumulator
from visualization import save_recon_grid, run_tsne_visualization


# -------------------------
# Utils
# -------------------------
def dataclass_from_dict(klass, dikt):
    """
    Recursively convert a dict into a (nested) dataclass instance.
    Supports:
    - nested dataclasses
    - Optional[T], Union
    - List[T], Dict[K,V]
    """
    if dikt is None:
        return None

    if not is_dataclass(klass):
        return dikt

    type_hints = get_type_hints(klass)
    kwargs = {}
    for f in fields(klass):
        name = f.name
        if name not in dikt:
            continue
        ft = type_hints.get(name, f.type)
        val = dikt[name]

        # Case 1: nested dataclass
        if is_dataclass(ft):
            kwargs[name] = dataclass_from_dict(ft, val)
            continue

        origin = get_origin(ft)
        args = get_args(ft)

        # Case 2: Optional / Union
        if origin is Union:
            non_none = [a for a in args if a is not type(None)]  # noqa: E721
            if len(non_none) == 1:
                inner = non_none[0]
                if is_dataclass(inner):
                    kwargs[name] = dataclass_from_dict(inner, val)
                else:
                    kwargs[name] = val
                continue

        # Case 3: list
        if origin in (list, list.__class__):
            inner = args[0] if args else Any
            if is_dataclass(inner):
                kwargs[name] = [dataclass_from_dict(inner, x) for x in val]
            else:
                kwargs[name] = val
            continue

        # Case 4: dict
        if origin in (dict, dict.__class__):
            kwargs[name] = val
            continue

        # default
        kwargs[name] = val

    return klass(**kwargs)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_checkpoint_any(path: str | Path, device: torch.device) -> Dict[str, Any]:
    p = Path(path)
    obj = torch.load(str(p), map_location=device)
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected checkpoint format: {path}")
    return obj


# -------------------------
# Losses (shared with trainer)
# -------------------------
try:
    # Package-style execution (for example: PYTHONPATH=src python -m phase1.eval)
    from phase1.training.recon_losses import (
        masked_bce_logits_weighted,
        mixed_bce_logits_weighted,
    )
except Exception:
    # Legacy execution from within src/phase1 (for example: python eval.py)
    from training.recon_losses import (
        masked_bce_logits_weighted,
        mixed_bce_logits_weighted,
    )


# -------------------------
# Visualization (same style as trainer)
# -------------------------
@torch.no_grad()
def visualize_once(
    out_dir: Path,
    step: int,
    x: torch.Tensor,
    recon_orig: torch.Tensor,
    recon_flip: torch.Tensor,
    pixel_mask: Optional[torch.Tensor],
    max_items: int = 8,
):
    ensure_dir(out_dir)
    save_recon_grid(
        out_path=str(out_dir / f"recon_{step:06d}.png"),
        x=x[:max_items],
        recon_orig=recon_orig[:max_items],
        recon_flip=recon_flip[:max_items],
        pixel_mask=None if pixel_mask is None else pixel_mask[:max_items],
    )


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def run_eval(
    model: SwinUnetDualViewSSL,
    loader: DataLoader,
    device: torch.device,
    cfg,
    out_dir: Path,
    visualize_every: int = 0,
) -> Dict[str, float]:
    model.eval()

    meter = MetricsAccumulator()
    loss_recon_sum = 0.0
    loss_con_sum = 0.0
    loss_var_sum = 0.0
    loss_total_sum = 0.0
    n_batches = 0

    for step, batch in enumerate(loader):
        x = batch["image"].to(device)  # [B, C, H, W]
        plane = batch.get("plane_one_hot", None)
        if plane is None:
            plane = torch.tensor([[0.0, 1.0]], device=device).repeat(x.size(0), 1)
        else:
            plane = plane.to(device)

        if cfg.mask.enable:
            from augmentation import sample_masks_anti_mirror

            pixel_mask = sample_masks_anti_mirror(
                x.shape,
                device=device,
                **cfg.mask.kwargs,
            )
        else:
            pixel_mask = None

        x_flip = flip_lr(x)

        out = model(
            x,
            plane,
            pixel_mask=pixel_mask,
            return_embeddings=cfg.training.enable_contrastive,
        )
        recon_raw_orig, recon_raw_flip, z1, z2, _ = out

        # Recon loss
        if cfg.training.recon_loss_type == "weighted_bce_logits":
            if cfg.training.enable_masked_loss:
                loss_orig = masked_bce_logits_weighted(recon_raw_orig, x, pixel_mask)
                loss_flip = masked_bce_logits_weighted(recon_raw_flip, x_flip, pixel_mask)
            else:
                loss_orig = mixed_bce_logits_weighted(recon_raw_orig, x, pixel_mask)
                loss_flip = mixed_bce_logits_weighted(recon_raw_flip, x_flip, pixel_mask)
        elif cfg.training.recon_loss_type == "l1_sigmoid":
            recon_orig = torch.sigmoid(recon_raw_orig).clamp(0, 1)
            recon_flip = torch.sigmoid(recon_raw_flip).clamp(0, 1)
            if cfg.training.enable_masked_loss:
                loss_orig = masked_l1_loss(recon_orig, x, pixel_mask)
                loss_flip = masked_l1_loss(recon_flip, x_flip, pixel_mask)
            else:
                loss_orig = mixed_l1_loss(recon_orig, x, pixel_mask)
                loss_flip = mixed_l1_loss(recon_flip, x_flip, pixel_mask)
        else:
            raise ValueError(f"Unknown recon_loss_type: {cfg.training.recon_loss_type}")

        loss_recon = 0.5 * (loss_orig + loss_flip)

        # Contrastive loss (if enabled)
        if cfg.training.enable_contrastive and z1 is not None and z2 is not None:
            loss_con = nt_xent_loss(
                z1,
                z2,
                temperature=cfg.training.contrastive_temperature,
            )
        else:
            loss_con = torch.tensor(0.0, device=device)

        # Variance reg (if enabled)
        if cfg.training.enable_variance_reg and z1 is not None and z2 is not None:
            loss_var = 0.5 * (
                variance_regularizer(z1, eps=cfg.training.variance_eps)
                + variance_regularizer(z2, eps=cfg.training.variance_eps)
            )
        else:
            loss_var = torch.tensor(0.0, device=device)

        loss_total = (
            cfg.training.recon_weight * loss_recon
            + cfg.training.contrastive_weight * loss_con
            + cfg.training.variance_weight * loss_var
        )

        # Metrics
        recon_orig_m = torch.sigmoid(recon_raw_orig).clamp(0, 1)
        recon_flip_m = torch.sigmoid(recon_raw_flip).clamp(0, 1)

        diff_orig = (recon_orig_m - x).abs()
        diff_flip = (recon_flip_m - x_flip).abs()
        diff_total = 0.5 * (diff_orig + diff_flip)

        ssim_orig = ssim_index(recon_orig_m, x)
        ssim_flip = ssim_index(recon_flip_m, x_flip)
        ssim_sum = 0.5 * (ssim_orig + ssim_flip)

        meter.update(diff_total, pixel_mask, ssim_sum=ssim_sum)

        # Accumulate losses
        loss_recon_sum += float(loss_recon.item())
        loss_con_sum += float(loss_con.item())
        loss_var_sum += float(loss_var.item())
        loss_total_sum += float(loss_total.item())
        n_batches += 1

        # Visualization
        if visualize_every and (step % visualize_every == 0):
            visualize_once(
                out_dir=out_dir,
                step=step,
                x=x,
                recon_orig=recon_orig_m,
                recon_flip=recon_flip_m,
                pixel_mask=pixel_mask,
                max_items=8,
            )

    metrics = meter.compute()
    metrics.update(
        {
            "loss_recon": loss_recon_sum / max(n_batches, 1),
            "loss_contrastive": loss_con_sum / max(n_batches, 1),
            "loss_variance": loss_var_sum / max(n_batches, 1),
            "loss_total": loss_total_sum / max(n_batches, 1),
        }
    )
    return metrics


# -------------------------
# Main
# -------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="eval_out")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--visualize_every", type=int, default=0)
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--tsne_max_items", type=int, default=2000)
    args = parser.parse_args()

    device = get_device()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Load checkpoint
    ckpt = load_checkpoint_any(args.ckpt, device=device)

    # Load cfg from checkpoint if available
    cfg_obj = ckpt.get("cfg", None)
    if isinstance(cfg_obj, dict):
        cfg = dataclass_from_dict(Config, cfg_obj)
    else:
        # Fallback: attempt to use cfg stored as dataclass already
        cfg = cfg_obj if cfg_obj is not None else Config()

    # Build dataset/loader
    tfm = get_eval_transforms(cfg)
    ds = FolderDataset(root=args.data_dir, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Build model
    model = SwinUnetDualViewSSL(cfg).to(device)

    # Load weights
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)

    # Run eval
    metrics = run_eval(
        model=model,
        loader=loader,
        device=device,
        cfg=cfg,
        out_dir=out_dir,
        visualize_every=int(args.visualize_every),
    )

    # Print and save
    print(json.dumps(metrics, indent=2))
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Optional tSNE
    if args.tsne:
        tsne_dir = out_dir / "tsne"
        ensure_dir(tsne_dir)

        # Wrapper: model returns embeddings
        class _Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, batch):
                x = batch["image"].to(device)
                plane = batch.get("plane_one_hot", None)
                if plane is None:
                    plane = torch.tensor([[0.0, 1.0]], device=device).repeat(x.size(0), 1)
                else:
                    plane = plane.to(device)

                if cfg.mask.enable:
                    from augmentation import sample_masks_anti_mirror

                    pixel_mask = sample_masks_anti_mirror(
                        x.shape,
                        device=device,
                        **cfg.mask.kwargs,
                    )
                else:
                    pixel_mask = None

                _, _, z1, _, _ = self.m(
                    x,
                    plane,
                    pixel_mask=pixel_mask,
                    return_embeddings=True,
                )
                return z1

        wrap = _Wrap(model)
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
