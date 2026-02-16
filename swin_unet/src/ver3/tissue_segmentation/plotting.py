from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _plot_two_lines(df: pd.DataFrame, x: str, y1: str, y2: str, out_path: Path, ylabel: str) -> None:
    if y1 not in df.columns or y2 not in df.columns:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(df[x], df[y1], label="train")
    plt.plot(df[x], df[y2], label="eval")
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

    _plot_two_lines(df, "epoch", "train_loss", "eval_loss", plot_dir / "loss.png", "loss")
    _plot_two_lines(df, "epoch", "train_macro_dice", "eval_macro_dice", plot_dir / "macro_dice.png", "macro_dice")
    _plot_two_lines(df, "epoch", "train_dice_min", "eval_dice_min", plot_dir / "dice_min.png", "dice_min")
    _plot_two_lines(df, "epoch", "train_dice_mean", "eval_dice_mean", plot_dir / "dice_mean.png", "dice_mean")
    _plot_two_lines(df, "epoch", "train_dice_max", "eval_dice_max", plot_dir / "dice_max.png", "dice_max")
    _plot_two_lines(
        df,
        "epoch",
        "train_num_valid_classes",
        "eval_num_valid_classes",
        plot_dir / "num_valid_classes.png",
        "num_valid_classes",
    )


__all__ = ["generate_plots"]
