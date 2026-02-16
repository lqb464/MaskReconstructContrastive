from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(10, 3 * n), squeeze=False)
    for i in range(n):
        ax0, ax1, ax2 = axes[i]
        ax0.imshow(x[i, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        ax0.set_title("input")
        ax0.axis("off")

        ax1.imshow(y[i].numpy(), cmap="tab20", vmin=0, vmax=max(1, num_classes - 1), interpolation="nearest")
        ax1.set_title("target")
        ax1.axis("off")

        ax2.imshow(pred[i].numpy(), cmap="tab20", vmin=0, vmax=max(1, num_classes - 1), interpolation="nearest")
        ax2.set_title("pred")
        ax2.axis("off")

    legend_items = []
    if class_names:
        for cid in sorted(class_names.keys())[: min(12, len(class_names))]:
            legend_items.append(f"{cid}:{class_names[cid]}")
    if legend_items:
        fig.suptitle(" | ".join(legend_items), fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


__all__ = ["save_val_visualization_grid"]
