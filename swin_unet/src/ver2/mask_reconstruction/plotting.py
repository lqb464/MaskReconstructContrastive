from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _plot_two_lines(df: pd.DataFrame, x: str, y1: str, y2: str, out_path: Path, ylabel: str):
    plt.figure(figsize=(6, 4))
    if y1 in df:
        plt.plot(df[x], df[y1], label="train")
    if y2 in df:
        plt.plot(df[x], df[y2], label="val")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_plots(csv_path: Path, plot_dir: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    plot_dir.mkdir(parents=True, exist_ok=True)

    _plot_two_lines(df, "epoch", "train_loss_total", "val_loss_total", plot_dir / "loss_total.png", "loss_total")
    _plot_two_lines(df, "epoch", "train_loss_masked", "val_loss_masked", plot_dir / "loss_masked.png", "loss_masked")
    _plot_two_lines(df, "epoch", "train_loss_unmasked", "val_loss_unmasked", plot_dir / "loss_unmasked.png", "loss_unmasked")
    _plot_two_lines(df, "epoch", "train_dice", "val_dice", plot_dir / "dice.png", "dice")
    if "train_loss_dice_aux" in df.columns and "val_loss_dice_aux" in df.columns:
        _plot_two_lines(df, "epoch", "train_loss_dice_aux", "val_loss_dice_aux", plot_dir / "dice_aux.png", "dice_aux")
    if "train_loss_contrastive" in df.columns and "val_loss_contrastive" in df.columns:
        _plot_two_lines(df, "epoch", "train_loss_contrastive", "val_loss_contrastive", plot_dir / "contrastive.png", "contrastive")


__all__ = ["generate_plots"]
