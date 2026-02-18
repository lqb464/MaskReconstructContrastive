from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _build_discrete_cmap(num_classes: int) -> tuple[ListedColormap, BoundaryNorm]:
    c = max(2, int(num_classes))
    base = plt.get_cmap("tab20").colors
    colors = [base[i % len(base)] for i in range(c)]
    cmap = ListedColormap(colors, name="tissue_classes")
    norm = BoundaryNorm(boundaries=[i - 0.5 for i in range(c + 1)], ncolors=c)
    return cmap, norm


def _present_class_ids(target: torch.Tensor, pred: torch.Tensor, num_classes: int) -> list[int]:
    ids = set(int(v) for v in torch.unique(target).tolist())
    ids.update(int(v) for v in torch.unique(pred).tolist())
    return [cid for cid in sorted(ids) if 0 <= cid < int(num_classes)]


def save_val_visualization_grid(
    *,
    val_batch: dict[str, torch.Tensor],
    out_path: Path,
    num_classes: int,
    class_names: Optional[Dict[int, str]] = None,
    max_items: int = 4,
) -> None:
    """
    Save a compact grid with columns: input | target | pred.
    """
    required = ("input", "target", "logits")
    missing = [k for k in required if k not in val_batch]
    if missing:
        raise RuntimeError(f"Visualization payload missing keys: {missing}")

    x = val_batch["input"]
    y = val_batch["target"]
    logits = val_batch["logits"]

    n = min(int(max_items), int(x.size(0)))
    if n <= 0:
        return

    x = x[:n].detach().cpu()
    y = y[:n].detach().cpu()
    pred = torch.argmax(logits[:n].detach().cpu(), dim=1)

    cmap, norm = _build_discrete_cmap(num_classes)
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(11, 3 * n), squeeze=False)
    for i in range(n):
        ax0, ax1, ax2 = axes[i]
        ax0.imshow(x[i, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        ax0.set_title("input")
        ax0.axis("off")

        ax1.imshow(y[i].numpy(), cmap=cmap, norm=norm, interpolation="nearest")
        ax1.set_title("target")
        ax1.axis("off")

        ax2.imshow(pred[i].numpy(), cmap=cmap, norm=norm, interpolation="nearest")
        ax2.set_title("pred")
        ax2.axis("off")

    present_ids = _present_class_ids(y, pred, num_classes)
    if present_ids:
        max_legend_items = 18
        legend_ids = present_ids[:max_legend_items]
        legend_handles = []
        for cid in legend_ids:
            name = class_names.get(cid, f"class_{cid}") if class_names else f"class_{cid}"
            legend_handles.append(Patch(facecolor=cmap(cid), edgecolor="black", label=f"{cid}: {name}"))
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(0.98, 0.5),
            frameon=True,
            fontsize=8,
            title="Classes (in view)",
            title_fontsize=9,
        )
        if len(present_ids) > max_legend_items:
            fig.text(
                0.98,
                0.05,
                f"... and {len(present_ids) - max_legend_items} more classes",
                ha="left",
                va="bottom",
                fontsize=8,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0.0, 0.0, 0.96, 1.0])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


__all__ = ["save_val_visualization_grid"]
