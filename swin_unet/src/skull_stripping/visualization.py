from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..models.swin_unet_dualview_ssl import flip_lr
from ..viz.visualization import (
    run_tsne_visualization as _run_tsne_visualization_base,
    save_image_grid as _save_image_grid_base,
)

log = logging.getLogger(__name__)

def save_image_grid(
    tensors: List[torch.Tensor],
    titles: List[str],
    out_path: str,
    annotations: Optional[Dict[int, List[str]]] = None,
    panel_vmax: Optional[Dict[int, float]] = None,
    max_items: int = 4,
) -> None:
    """
    Thin wrapper with hard item cap to avoid unbounded visualization memory.
    """
    max_items = max(1, int(max_items))
    capped_tensors = [t[:max_items] for t in tensors]
    _save_image_grid_base(
        tensors=capped_tensors,
        titles=titles,
        out_path=out_path,
        annotations=annotations,
        panel_vmax=panel_vmax,
    )
    del capped_tensors

def run_tsne_visualization(*, enabled: bool = False, **kwargs):
    if not enabled:
        return None
    log.warning("t-SNE visualization enabled; this is CPU heavy.")
    return _run_tsne_visualization_base(**kwargs)

@torch.no_grad()
def save_val_visualization_grid(
    *,
    val_batch: dict[str, torch.Tensor],
    device: torch.device,
    out_path: Path,
    threshold: float,
    max_items: int,
    dice_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    show_flip: bool = True,
    compact_mode: bool = True,
    debug_overlay: bool = False,
    debug_unletterbox: bool = False,
) -> None:
    """
    Validation visualization with dual-view support.
    This function only consumes precomputed tensors from validation and never performs model forward.
    """
    max_items = min(int(max_items), 4)

    if max_items <= 0:
        return

    required_keys = ("input", "target", "recon1_logits")
    missing_required = [k for k in required_keys if k not in val_batch]
    if missing_required:
        raise RuntimeError(
            "save_val_visualization_grid requires precomputed validation outputs. "
            f"Missing keys: {missing_required}. Pass recon logits from validation forward."
        )

    x = val_batch["input"].to(device, non_blocking=True)
    y = val_batch["target"].to(device, non_blocking=True)

    n_items = min(max_items, x.size(0))
    if n_items <= 0:
        return

    x = x[:n_items]
    y = y[:n_items]
    pixel_mask = val_batch.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask[:n_items].to(device, non_blocking=True)
        mask_coverage = float(pixel_mask.float().mean().item())
        print(f"[vis] masking enabled in visualization payload, mask_coverage={mask_coverage:.4f}")
    else:
        print("[vis] masking not present in visualization payload.")

    x_display = x if pixel_mask is None else (x * (1.0 - pixel_mask))
    y_flip = val_batch.get("target_flip")
    if y_flip is not None:
        y_flip = y_flip[:n_items].to(device, non_blocking=True)
    else:
        y_flip = flip_lr(y)

    logits_orig = val_batch["recon1_logits"][:n_items].to(device, non_blocking=True)

    logits_flip = None
    if show_flip:
        if "recon2_logits" not in val_batch:

            print("[vis] show_flip requested but recon2_logits is missing; fallback to single-view visualization.")
            show_flip = False
        else:
            logits_flip = val_batch["recon2_logits"][:n_items].to(device, non_blocking=True)

    prob_orig = torch.sigmoid(logits_orig)
    bin_orig = (prob_orig >= threshold).float()
    x_flip_raw = torch.flip(x, dims=[-1]) if show_flip else None
    x_flip_display = x_flip_raw
    if show_flip and x_flip_display is not None and pixel_mask is not None:
        x_flip_display = x_flip_display * (1.0 - pixel_mask)
    if show_flip and logits_flip is not None:
        prob_flip = torch.sigmoid(logits_flip)
        bin_flip = (prob_flip >= threshold).float()
    else:
        prob_flip = None
        bin_flip = None

    samples = []
    for i in range(n_items):
        dice_o = dice_fn(prob_orig[i : i + 1], y[i : i + 1]).item()
        dice_f = dice_fn(prob_flip[i : i + 1], y_flip[i : i + 1]).item() if show_flip and prob_flip is not None else None
        samples.append(
            {
                "x": x[i : i + 1].cpu(),
                "x_masked": x_display[i : i + 1].cpu(),
                "y": y[i : i + 1].cpu(),
                "prob_o": prob_orig[i : i + 1].cpu(),
                "bin_o": bin_orig[i : i + 1].cpu(),
                "mask": pixel_mask[i : i + 1].cpu() if pixel_mask is not None else None,
                "x_flip": x_flip_display[i : i + 1].cpu() if show_flip and x_flip_display is not None else None,
                "prob_f": prob_flip[i : i + 1].cpu() if show_flip and prob_flip is not None else None,
                "bin_f": bin_flip[i : i + 1].cpu() if show_flip and bin_flip is not None else None,
                "dice_o": dice_o,
                "dice_f": dice_f,
            }
        )

    def _save_overlay(x_in: torch.Tensor, y_in: torch.Tensor, path: Path, k: int = 4):
        os.makedirs(path.parent, exist_ok=True)
        k = min(k, x_in.size(0))
        fig, axes = plt.subplots(1, k, figsize=(3 * k, 3))
        if k == 1:
            axes = [axes]
        for i in range(k):
            ax = axes[i]
            img = x_in[i, 0].cpu().numpy()
            mask = y_in[i, 0].cpu().numpy()
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.imshow(mask, cmap="autumn", alpha=0.35, vmin=0, vmax=max(1.0, mask.max()))
            ax.contour(mask, levels=[0.5], colors="lime", linewidths=1.0)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)

    def _tight_bbox(mask: torch.Tensor):
        nz = (mask > 0).nonzero(as_tuple=False)
        if nz.numel() == 0:
            return None
        rmin, cmin = nz[:, 1].min().item(), nz[:, 2].min().item()
        rmax, cmax = nz[:, 1].max().item(), nz[:, 2].max().item()
        return rmin, rmax, cmin, cmax

    def _save_cropped(x_in, y_in, prob_o, bin_o, prob_f, bin_f, path: Path, out_size: int = 256):
        os.makedirs(path.parent, exist_ok=True)
        bbox = _tight_bbox(y_in)
        if bbox is None:
            return
        rmin, rmax, cmin, cmax = bbox
        margin = 8
        r0 = max(0, rmin - margin)
        r1 = min(y_in.shape[-2], rmax + margin + 1)
        c0 = max(0, cmin - margin)
        c1 = min(y_in.shape[-1], cmax + margin + 1)

        def crop_and_resize(t: torch.Tensor, *, is_mask: bool) -> torch.Tensor:
            t = t[:, :, r0:r1, c0:c1]
            if is_mask:
                return F.interpolate(t.float(), size=(out_size, out_size), mode="nearest")
            return F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)

        tensors = [crop_and_resize(x_in, is_mask=False), crop_and_resize(y_in, is_mask=True)]
        titles = ["input_crop", "target_crop"]
        annotations = {}

        tensors += [crop_and_resize(prob_o, is_mask=False), crop_and_resize(bin_o, is_mask=True)]
        titles += ["prob_o_crop", "bin_o_crop"]
        if prob_f is not None and bin_f is not None:
            tensors += [crop_and_resize(prob_f, is_mask=False), crop_and_resize(bin_f, is_mask=True)]
            titles += ["prob_f_crop", "bin_f_crop"]

        save_image_grid(tensors=tensors, titles=titles, out_path=str(path), annotations=annotations, max_items=max_items)
        del tensors, titles, annotations

    def _pack_and_save(items, path: Path):
        if len(items) == 0:
            return
        x_cat = torch.cat([s["x_masked"] for s in items], dim=0)
        y_cat = torch.cat([s["y"] for s in items], dim=0)
        prob_o = torch.cat([s["prob_o"] for s in items], dim=0)
        bin_o = torch.cat([s["bin_o"] for s in items], dim=0)
        ann_o = [f"Dice(o)={s['dice_o']:.3f}" for s in items]

        tensors = [x_cat]
        titles = ["input_masked" if items[0]["mask"] is not None else "input"]
        if items[0]["mask"] is not None:
            mask_cat = torch.cat([s["mask"] for s in items], dim=0)
            tensors += [mask_cat]
            titles += ["pixel_mask"]
        tensors += [y_cat]
        titles += ["target"]
        annotations: dict[int, list[str]] = {}

        if not compact_mode:
            tensors += [prob_o]
            titles += ["pred_prob_o"]
            col_bin_o = len(tensors)
            tensors += [bin_o]
            titles += [f"pred_bin_o(th={threshold})"]
        else:
            col_bin_o = len(tensors)
            tensors += [bin_o]
            titles += [f"pred_bin_o(th={threshold})"]
        annotations[col_bin_o] = ann_o

        if show_flip and items[0]["x_flip"] is not None and items[0]["prob_f"] is not None and items[0]["bin_f"] is not None:
            x_f = torch.cat([s["x_flip"] for s in items], dim=0)
            prob_f = torch.cat([s["prob_f"] for s in items], dim=0)
            bin_f = torch.cat([s["bin_f"] for s in items], dim=0)
            ann_f = [f"Dice(f)={s['dice_f']:.3f}" for s in items]

            tensors += [x_f]
            titles += ["input_flip_masked" if items[0]["mask"] is not None else "input_flip"]
            if not compact_mode:
                tensors += [prob_f]
                titles += ["pred_prob_f"]
                col_bin_f = len(tensors)
                tensors += [bin_f]
                titles += [f"pred_bin_f(th={threshold})"]
            else:
                col_bin_f = len(tensors)
                tensors += [bin_f]
                titles += [f"pred_bin_f(th={threshold})"]
            annotations[col_bin_f] = [f"{a_o} | {a_f}" for a_o, a_f in zip(ann_o, ann_f)]
        else:
            prob_f = None
            bin_f = None

        save_image_grid(tensors=tensors, titles=titles, out_path=str(path), annotations=annotations, max_items=max_items)

        if debug_overlay:
            _save_overlay(x_cat, y_cat, path.with_name(path.stem + "_overlay.png"))
        if debug_unletterbox:
            _save_cropped(
                x_cat,
                y_cat,
                prob_o,
                bin_o,
                prob_f if show_flip else None,
                bin_f if show_flip else None,
                path.with_name(path.stem + "_cropped.png"),
            )

        del x_cat, y_cat, prob_o, bin_o, tensors, titles, annotations, ann_o
        if items[0]["mask"] is not None:
            del mask_cat
        if prob_f is not None:
            del prob_f
        if bin_f is not None:
            del bin_f

    if len(samples) == 0:
        return

    _pack_and_save(samples, out_path)
    del samples, x, y, logits_orig, prob_orig, bin_orig
    if show_flip:
        del y_flip
        if x_flip_raw is not None:
            del x_flip_raw
        if x_flip_display is not None:
            del x_flip_display
        if logits_flip is not None:
            del logits_flip
        if prob_flip is not None:
            del prob_flip
        if bin_flip is not None:
            del bin_flip

__all__ = ["save_val_visualization_grid", "save_image_grid", "run_tsne_visualization"]
