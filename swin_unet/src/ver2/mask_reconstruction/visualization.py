from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..viz.visualization import save_image_grid


@torch.no_grad()
def save_val_visualization_grid(
    *,
    model,
    val_loader,
    device: torch.device,
    out_path: Path,
    threshold: float,
    max_items: int,
    dice_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    max_batches: int = 4,
    items_per_batch: int | None = None,
    show_flip: bool = True,
    compact_mode: bool = True,
    save_per_batch: bool = False,
    debug_overlay: bool = False,
    debug_unletterbox: bool = False,
) -> None:
    """
    Validation visualization with dual-view support.
    Collects samples from multiple batches (up to max_batches, max_items).
    If save_per_batch=True, writes one PNG per batch with suffix _bXX.
    """
    model_was_training = model.training
    model.eval()

    def _run_batch(batch):
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        plane = batch["plane_one_hot"].to(device, non_blocking=True)

        pixel_mask = torch.zeros_like(y)
        logits_orig, _, _, _ = model(x, pixel_mask, plane)
        prob_orig = torch.sigmoid(logits_orig)
        bin_orig = (prob_orig >= threshold).float()

        if show_flip:
            x_flip = torch.flip(x, dims=[-1])
            logits_flip, _, _, _ = model(x_flip, pixel_mask, plane)
            prob_flip = torch.sigmoid(logits_flip)
            bin_flip = (prob_flip >= threshold).float()
        else:
            x_flip = None
            prob_flip = None
            bin_flip = None

        # Dice per sample
        for i in range(x.size(0)):
            dice_o = dice_fn(prob_orig[i : i + 1], y[i : i + 1]).item()
            dice_f = dice_fn(prob_flip[i : i + 1], y[i : i + 1]).item() if show_flip else None
            yield {
                "x": x[i : i + 1].cpu(),
                "y": y[i : i + 1].cpu(),
                "prob_o": prob_orig[i : i + 1].cpu(),
                "bin_o": bin_orig[i : i + 1].cpu(),
                "x_flip": x_flip[i : i + 1].cpu() if show_flip else None,
                "prob_f": prob_flip[i : i + 1].cpu() if show_flip else None,
                "bin_f": bin_flip[i : i + 1].cpu() if show_flip else None,
                "dice_o": dice_o,
                "dice_f": dice_f,
            }

    def _save_overlay(x_in: torch.Tensor, y_in: torch.Tensor, path: Path, k: int = 4):
        """Save overlay of mask on input for first k samples."""
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
            # boundary
            from matplotlib import patches

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
        """Center crop to mask bbox (with margin) then resize for display only."""
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

        def crop_and_resize(t):
            t = t[:, :, r0:r1, c0:c1]
            return torch.nn.functional.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False) if t.dtype == torch.float32 else torch.nn.functional.interpolate(t.float(), size=(out_size, out_size), mode="nearest")

        tensors = [crop_and_resize(x_in), crop_and_resize(y_in)]
        titles = ["input_crop", "target_crop"]
        annotations = {}

        tensors += [crop_and_resize(prob_o), crop_and_resize(bin_o)]
        titles += ["prob_o_crop", "bin_o_crop"]
        if prob_f is not None and bin_f is not None:
            tensors += [crop_and_resize(prob_f), crop_and_resize(bin_f)]
            titles += ["prob_f_crop", "bin_f_crop"]

        save_image_grid(tensors=tensors, titles=titles, out_path=str(path), annotations=annotations)

    def _pack_and_save(samples, path: Path):
        if len(samples) == 0:
            return
        x_cat = torch.cat([s["x"] for s in samples], dim=0)
        y_cat = torch.cat([s["y"] for s in samples], dim=0)
        prob_o = torch.cat([s["prob_o"] for s in samples], dim=0)
        bin_o = torch.cat([s["bin_o"] for s in samples], dim=0)
        ann_o = [f"Dice(o)={s['dice_o']:.3f}" for s in samples]

        tensors = [x_cat, y_cat]
        titles = ["input", "target"]
        annotations = {}

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

        if show_flip:
            x_f = torch.cat([s["x_flip"] for s in samples], dim=0)
            prob_f = torch.cat([s["prob_f"] for s in samples], dim=0)
            bin_f = torch.cat([s["bin_f"] for s in samples], dim=0)
            ann_f = [f"Dice(f)={s['dice_f']:.3f}" for s in samples]

            tensors += [x_f]
            titles += ["input_flip"]
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

        save_image_grid(tensors=tensors, titles=titles, out_path=str(path), annotations=annotations)

        if debug_overlay:
            _save_overlay(x_cat, y_cat, path.with_name(path.stem + "_overlay.png"))
        if debug_unletterbox:
            _save_cropped(x_cat, y_cat, prob_o, bin_o, prob_f if show_flip else None, bin_f if show_flip else None, path.with_name(path.stem + "_cropped.png"))

    samples = []
    batch_count = 0
    for b_idx, batch in enumerate(val_loader):
        batch_count += 1
        per_batch_items = items_per_batch or batch["input"].size(0)
        batch_samples = []
        for k, s in enumerate(_run_batch(batch)):
            if k >= per_batch_items:
                break
            batch_samples.append(s)
            if not save_per_batch:
                samples.append(s)
            if not save_per_batch and len(samples) >= max_items:
                break
        if save_per_batch and len(batch_samples) > 0:
            _pack_and_save(batch_samples, out_path.with_name(out_path.stem + f"_b{b_idx:02d}.png"))
        if (not save_per_batch and len(samples) >= max_items) or batch_count >= max_batches:
            break

    if save_per_batch:
        if model_was_training:
            model.train()
        return

    if len(samples) == 0:
        if model_was_training:
            model.train()
        return

    # compact concatenation
    samples = samples[:max_items]
    _pack_and_save(samples, out_path)

    if model_was_training:
        model.train()


__all__ = ["save_val_visualization_grid"]
