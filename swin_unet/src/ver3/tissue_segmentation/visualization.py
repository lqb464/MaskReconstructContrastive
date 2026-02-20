from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib
import torch
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_CLASS_NAMES = [
    "Background",
    "Gray Matter",
    "White Matter",
    "CSF",
    "Skull",
    "Other",
]

# Fixed categorical colors aligned by class index.
DEFAULT_CLASS_COLORS = [
    "#000000",  # Background
    "#d62728",  # Gray Matter
    "#1f77b4",  # White Matter
    "#17becf",  # CSF
    "#ff7f0e",  # Skull
    "#2ca02c",  # Other
]


def _resolve_class_names(
    class_names: Optional[Dict[int, str] | Sequence[str]],
    num_classes: int,
) -> list[str]:
    c = int(num_classes)
    if c <= 0:
        raise ValueError(f"num_classes must be > 0, got {c}")

    if class_names is None:
        names = list(DEFAULT_CLASS_NAMES)
    elif isinstance(class_names, dict):
        names = [str(class_names.get(i, f"class_{i}")) for i in range(c)]
    else:
        names = [str(x) for x in class_names]

    if len(names) < c:
        names = names + [f"class_{i}" for i in range(len(names), c)]
    elif len(names) > c:
        names = names[:c]
    return names


def _build_segmentation_colormap(num_classes: int) -> ListedColormap:
    c = int(num_classes)
    if c <= 0:
        raise ValueError(f"num_classes must be > 0, got {c}")

    if c <= len(DEFAULT_CLASS_COLORS):
        colors = DEFAULT_CLASS_COLORS[:c]
    else:
        # Keep first six fixed, extend deterministically for larger class counts.
        extra = c - len(DEFAULT_CLASS_COLORS)
        tab = plt.get_cmap("tab20").colors
        colors = list(DEFAULT_CLASS_COLORS) + [tab[i % len(tab)] for i in range(extra)]
    return ListedColormap(colors, name="tissue_classes")


def create_segmentation_legend(
    ax: plt.Axes,
    class_ids: Sequence[int],
    class_names: Sequence[str],
    colormap: ListedColormap,
) -> None:
    handles = [
        mpatches.Patch(facecolor=colormap(int(cid)), edgecolor="black", label=str(class_names[int(cid)]))
        for cid in class_ids
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=True,
        title="Classes",
        fontsize=8,
        title_fontsize=9,
        borderaxespad=0.0,
    )
    ax.axis("off")


def _present_class_ids(target: torch.Tensor, pred: torch.Tensor) -> list[int]:
    ids = set(int(v) for v in torch.unique(target).tolist())
    ids.update(int(v) for v in torch.unique(pred).tolist())
    return [cid for cid in sorted(ids) if cid >= 0]


def _identified_class_ids(
    class_names: Optional[Dict[int, str] | Sequence[str]],
    num_classes: int,
) -> list[int]:
    c = int(num_classes)
    if c <= 0:
        return []
    if isinstance(class_names, dict):
        ids: set[int] = set()
        for raw_k in class_names.keys():
            try:
                cid = int(raw_k)
            except Exception:
                continue
            if 0 <= cid < c:
                ids.add(cid)
        return sorted(ids)
    return list(range(c))


def _per_class_dice_for_display(
    target: torch.Tensor,
    pred: torch.Tensor,
    class_ids: Sequence[int],
    *,
    eps: float = 1e-6,
) -> dict[int, float | None]:
    """
    Compute per-class Dice on visualization tensors.
    Returns None for classes with zero denominator (absent in both pred and target).
    """
    out: dict[int, float | None] = {}
    for cid in class_ids:
        tgt_c = (target == int(cid))
        pred_c = (pred == int(cid))
        tgt_sum = int(tgt_c.sum().item())
        pred_sum = int(pred_c.sum().item())
        denom = tgt_sum + pred_sum
        if denom == 0:
            out[int(cid)] = None
            continue
        inter = int((tgt_c & pred_c).sum().item())
        out[int(cid)] = float((2.0 * inter + eps) / (denom + eps))
    return out


def save_val_visualization_grid(
    *,
    val_batch: dict[str, torch.Tensor],
    out_path: Path,
    num_classes: int,
    class_names: Optional[Dict[int, str] | Sequence[str]] = None,
    max_items: int = 4,
    flip_horizontal: bool = False,
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
    if flip_horizontal:
        x = torch.flip(x, dims=[-1])
        y = torch.flip(y, dims=[-1])
        pred = torch.flip(pred, dims=[-1])

    names = _resolve_class_names(class_names, num_classes)
    cmap = _build_segmentation_colormap(num_classes)
    norm = BoundaryNorm(np.arange(-0.5, int(num_classes) + 0.5, 1.0), cmap.N)

    present_ids = _present_class_ids(y, pred)
    if present_ids:
        observed_max = int(max(present_ids))
        if observed_max >= int(num_classes):
            raise ValueError(
                f"Found label id {observed_max} >= num_classes ({num_classes}) in visualization payload."
            )
        if observed_max >= len(names):
            raise ValueError(
                f"Found label id {observed_max} but class_names has length {len(names)}."
            )

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(11, 3 * n), squeeze=False)
    for i in range(n):
        ax0, ax1, ax2 = axes[i]
        ax0.imshow(x[i, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        ax0.set_title("input (flip)" if flip_horizontal else "input")
        ax0.axis("off")

        ax1.imshow(y[i].numpy(), cmap=cmap, norm=norm, interpolation="nearest")
        ax1.set_title("target (flip)" if flip_horizontal else "target")
        ax1.axis("off")

        ax2.imshow(pred[i].numpy(), cmap=cmap, norm=norm, interpolation="nearest")
        ax2.set_title("pred (flip)" if flip_horizontal else "pred")
        ax2.axis("off")

    # Single legend for the whole figure, outside the image grid.
    identified_ids = _identified_class_ids(class_names, num_classes)
    identified_set = set(identified_ids)
    legend_ids = identified_ids
    if not legend_ids:
        legend_ids = list(range(int(num_classes)))

    dice_map = _per_class_dice_for_display(y, pred, legend_ids)

    legend_ax = fig.add_axes([0.83, 0.12, 0.16, 0.76])
    indexed_names = [f"{cid}: {names[cid]}" for cid in range(len(names))]
    for cid in legend_ids:
        d = dice_map.get(int(cid), None)
        if d is None:
            indexed_names[int(cid)] = f"{int(cid)}: {names[int(cid)]} (dice=NA)"
        else:
            indexed_names[int(cid)] = f"{int(cid)}: {names[int(cid)]} (dice={d:.3f})"
    create_segmentation_legend(legend_ax, legend_ids, indexed_names, cmap)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Avoid tight_layout with manually-added Axes (legend_ax), which triggers warnings.
    fig.subplots_adjust(left=0.03, right=0.81, top=0.96, bottom=0.04, wspace=0.08, hspace=0.22)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


__all__ = ["create_segmentation_legend", "save_val_visualization_grid"]
