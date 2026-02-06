from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from viz.visualization import save_image_grid


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
) -> None:
    """
    Run a small validation visualization pass and save a PNG grid.
    Uses view1 outputs for display.
    """
    model_was_training = model.training
    model.eval()

    xs = []
    targets = []
    preds_prob = []
    preds_bin = []
    ann = []

    collected = 0
    for batch in val_loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        plane = batch["plane_one_hot"].to(device, non_blocking=True)

        pixel_mask = torch.zeros_like(y)
        logits1, _, _, _ = model(x, pixel_mask, plane)
        prob1 = torch.sigmoid(logits1)
        bin1 = (prob1 >= threshold).float()

        # Dice per sample
        for i in range(x.size(0)):
            dice_val = dice_fn(prob1[i : i + 1], y[i : i + 1]).item()
            ann.append(f"Dice={dice_val:.3f}")

        xs.append(x.cpu())
        targets.append(y.cpu())
        preds_prob.append(prob1.cpu())
        preds_bin.append(bin1.cpu())

        collected += x.size(0)
        if collected >= max_items:
            break

    if len(xs) == 0:
        if model_was_training:
            model.train()
        return

    x_cat = torch.cat(xs, dim=0)[:max_items]
    y_cat = torch.cat(targets, dim=0)[:max_items]
    p_cat = torch.cat(preds_prob, dim=0)[:max_items]
    b_cat = torch.cat(preds_bin, dim=0)[:max_items]
    ann = ann[:max_items]

    annotations = {2: ann, 3: ann}
    titles = ["input", "target", "pred_prob", f"pred_bin(th={threshold})"]

    save_image_grid(
        tensors=[x_cat, y_cat, p_cat, b_cat],
        titles=titles,
        out_path=str(out_path),
        annotations=annotations,
    )

    if model_was_training:
        model.train()


__all__ = ["save_val_visualization_grid"]
