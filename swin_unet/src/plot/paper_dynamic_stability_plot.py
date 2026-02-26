# python swin_unet/src/plot/paper_dynamic_stability_plot.py \
#   --baseline-csv swin_unet/outputs/ixi/t1-axial/0.5/csv/no_saca.csv \
#   --our-csv swin_unet/outputs/ixi/t1-axial/0.5/csv/multi_saca.csv \
#   --out-dir swin_unet/outputs/ixi/t1-axial/0.5/csv/figures \
#   --baseline-color 505050 \
#   --our-color 0000ff \
#   --train-alpha 0.35 \
#   --val-alpha 0.95 \
#   --train-lighten 0.4

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _normalize_hex_color(color: str) -> str:
    value = color.strip()
    if not value.startswith("#"):
        value = f"#{value}"
    if len(value) != 7:
        raise ValueError(f"Invalid color '{color}'. Expected RRGGBB or #RRGGBB.")
    return value


def _lighten_hex(color: str, amount: float = 0.3) -> tuple[float, float, float]:
    color = color.lstrip("#")
    r = int(color[0:2], 16) / 255.0
    g = int(color[2:4], 16) / 255.0
    b = int(color[4:6], 16) / 255.0
    amount = max(0.0, min(1.0, amount))
    return (r + (1.0 - r) * amount, g + (1.0 - g) * amount, b + (1.0 - b) * amount)


def _load_curve(csv_path: Path, max_epoch: int = 150) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        raise ValueError(f"'epoch' column not found in {csv_path}")
    df = df[df["epoch"] <= max_epoch].copy()
    return df


def _style_minimal_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Keep tick marks, but hide all tick label text/numbers.
    ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, labeltop=False, labelright=False)


def _plot_metric(
    *,
    baseline_df: pd.DataFrame,
    our_df: pd.DataFrame,
    train_col: str,
    val_col: str,
    y_label: str,
    y_min: float,
    y_max: float,
    y_step: float,
    baseline_color: str,
    our_color: str,
    train_alpha: float,
    val_alpha: float,
    train_lighten: float,
    out_full: Path,
    out_minimal: Path,
) -> None:
    baseline_train_color = _lighten_hex(baseline_color, amount=train_lighten)
    our_train_color = _lighten_hex(our_color, amount=train_lighten)

    # Full version
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw train first (dotted + faded), then val to avoid overlap over val.
    ax.plot(
        baseline_df["epoch"],
        baseline_df[train_col],
        linestyle=":",
        linewidth=2.0,
        alpha=train_alpha,
        color=baseline_train_color,
        label="Baseline Train",
        zorder=1,
    )
    ax.plot(
        our_df["epoch"],
        our_df[train_col],
        linestyle=":",
        linewidth=2.0,
        alpha=train_alpha,
        color=our_train_color,
        label="Our Model Train",
        zorder=1,
    )
    ax.plot(
        baseline_df["epoch"],
        baseline_df[val_col],
        linestyle="-",
        linewidth=2.2,
        alpha=val_alpha,
        color=baseline_color,
        label="Baseline Val",
        zorder=2,
    )
    ax.plot(
        our_df["epoch"],
        our_df[val_col],
        linestyle="-",
        linewidth=2.2,
        alpha=val_alpha,
        color=our_color,
        label="Our Model Val",
        zorder=2,
    )

    ax.set_xlim(0, 150)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_ylim(y_min, y_max)
    y_ticks = []
    v = y_min
    while v <= y_max + 1e-9:
        y_ticks.append(round(v, 10))
        v += y_step
    ax.set_yticks(y_ticks)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_full.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_full, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Minimal version (no axis titles/ticks/legend)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        baseline_df["epoch"],
        baseline_df[train_col],
        linestyle=":",
        linewidth=2.0,
        alpha=train_alpha,
        color=baseline_train_color,
        zorder=1,
    )
    ax.plot(
        our_df["epoch"],
        our_df[train_col],
        linestyle=":",
        linewidth=2.0,
        alpha=train_alpha,
        color=our_train_color,
        zorder=1,
    )
    ax.plot(
        baseline_df["epoch"],
        baseline_df[val_col],
        linestyle="-",
        linewidth=2.2,
        alpha=val_alpha,
        color=baseline_color,
        zorder=2,
    )
    ax.plot(
        our_df["epoch"],
        our_df[val_col],
        linestyle="-",
        linewidth=2.2,
        alpha=val_alpha,
        color=our_color,
        zorder=2,
    )
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)
    _style_minimal_axes(ax)
    fig.tight_layout()
    out_minimal.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_minimal, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_dynamic_stability_plots(
    *,
    baseline_csv: Path,
    our_csv: Path,
    out_dir: Path,
    baseline_color: str = "505050",
    our_color: str = "0000ff",
    train_alpha: float = 0.5,
    val_alpha: float = 1.0,
    train_lighten: float = 0.3,
) -> None:
    baseline_color = _normalize_hex_color(baseline_color)
    our_color = _normalize_hex_color(our_color)

    baseline_df = _load_curve(baseline_csv, max_epoch=150)
    our_df = _load_curve(our_csv, max_epoch=150)

    _plot_metric(
        baseline_df=baseline_df,
        our_df=our_df,
        train_col="train_loss",
        val_col="val_loss",
        y_label="Total Loss",
        y_min=6.25,
        y_max=7.00,
        y_step=0.25,
        baseline_color=baseline_color,
        our_color=our_color,
        train_alpha=train_alpha,
        val_alpha=val_alpha,
        train_lighten=train_lighten,
        out_full=out_dir / "dynamic_total_loss_full.svg",
        out_minimal=out_dir / "dynamic_total_loss_minimal.svg",
    )
    _plot_metric(
        baseline_df=baseline_df,
        our_df=our_df,
        train_col="train_ssim",
        val_col="val_ssim",
        y_label="SSIM",
        y_min=0.50,
        y_max=1.00,
        y_step=0.25,
        baseline_color=baseline_color,
        our_color=our_color,
        train_alpha=train_alpha,
        val_alpha=val_alpha,
        train_lighten=train_lighten,
        out_full=out_dir / "dynamic_ssim_full.svg",
        out_minimal=out_dir / "dynamic_ssim_minimal.svg",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot dynamic/stability curves for baseline vs our model.")
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("swin_unet/outputs/ixi/t1-axial/0.5/csv/no_saca.csv"),
    )
    parser.add_argument(
        "--our-csv",
        type=Path,
        default=Path("swin_unet/outputs/ixi/t1-axial/0.5/csv/multi_saca.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("swin_unet/outputs/ixi/t1-axial/0.5/csv/figures"),
    )
    parser.add_argument("--baseline-color", type=str, default="505050", help="Hex color, e.g. 505050 or #505050")
    parser.add_argument("--our-color", type=str, default="0000ff", help="Hex color, e.g. 0000ff or #0000ff")
    parser.add_argument("--train-alpha", type=float, default=0.5, help="Alpha for train curves (default: 0.5)")
    parser.add_argument("--val-alpha", type=float, default=1.0, help="Alpha for val curves (default: 1.0)")
    parser.add_argument(
        "--train-lighten",
        type=float,
        default=0.3,
        help="Lighten factor for train colors in [0,1] (default: 0.3)",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    build_dynamic_stability_plots(
        baseline_csv=args.baseline_csv,
        our_csv=args.our_csv,
        out_dir=args.out_dir,
        baseline_color=args.baseline_color,
        our_color=args.our_color,
        train_alpha=args.train_alpha,
        val_alpha=args.val_alpha,
        train_lighten=args.train_lighten,
    )


if __name__ == "__main__":
    main()
