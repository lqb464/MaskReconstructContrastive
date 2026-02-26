from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _normalize_hex_color(color: str) -> str:
    value = str(color).strip()
    if not value.startswith("#"):
        value = f"#{value}"
    if len(value) != 7:
        raise ValueError(f"Invalid color '{color}'. Expected RRGGBB or #RRGGBB.")
    return value


def _load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    portion_col = "partion" if "partion" in df.columns else "portion"
    required = {
        portion_col,
        "train_loss_total",
        "train_dice",
        "train_dice_boundary",
        "train_dice_interior",
        "val_loss_total",
        "val_dice",
        "val_dice_boundary",
        "val_dice_interior",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Invalid schema in {csv_path}. Missing columns: {sorted(missing)}")

    numeric_cols = list(required)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols).copy()
    if df.empty:
        raise ValueError(f"No valid numeric rows in {csv_path}")

    df = df.sort_values(portion_col, ascending=True).reset_index(drop=True)
    if portion_col != "partion":
        df["partion"] = df[portion_col]
    return df


def _apply_x_axis_style(ax: plt.Axes, x_vals, hide_x_axis: bool) -> None:
    if hide_x_axis:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xlabel("partion")
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{v:g}" for v in x_vals])


def plot_loss_vs_partion(
    df: pd.DataFrame,
    out_path: Path,
    train_loss_color: str,
    val_loss_color: str,
    dpi: int,
    hide_x_axis: bool,
) -> None:
    x = df["partion"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        x,
        df["train_loss_total"],
        linestyle=":",
        marker="o",
        color=train_loss_color,
        linewidth=2.0,
        label="train_loss_total",
    )
    ax.plot(
        x,
        df["val_loss_total"],
        linestyle="-",
        marker="o",
        color=val_loss_color,
        linewidth=2.0,
        label="val_loss_total",
    )
    _apply_x_axis_style(ax, x, hide_x_axis)
    ax.set_ylabel("loss")
    ax.set_title("Mask Reconstruction: Loss vs Partion")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def plot_main_dice_vs_partion(
    df: pd.DataFrame,
    out_path: Path,
    train_dice_color: str,
    val_dice_color: str,
    dpi: int,
    hide_x_axis: bool,
) -> None:
    x = df["partion"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        x,
        df["train_dice"],
        linestyle=":",
        marker="s",
        color=train_dice_color,
        linewidth=2.0,
        label="train_dice",
    )
    ax.plot(
        x,
        df["val_dice"],
        linestyle="-",
        marker="s",
        color=val_dice_color,
        linewidth=2.0,
        label="val_dice",
    )
    _apply_x_axis_style(ax, x, hide_x_axis)
    ax.set_ylabel("dice")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Mask Reconstruction: Dice vs Partion")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def plot_other_dice_vs_partion(
    df: pd.DataFrame,
    out_path: Path,
    train_boundary_color: str,
    val_boundary_color: str,
    train_interior_color: str,
    val_interior_color: str,
    dpi: int,
    hide_x_axis: bool,
) -> None:
    x = df["partion"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        x,
        df["train_dice_boundary"],
        linestyle=":",
        marker="^",
        color=train_boundary_color,
        linewidth=2.0,
        alpha=0.85,
        label="train_dice_boundary",
    )
    ax.plot(
        x,
        df["val_dice_boundary"],
        linestyle="-",
        marker="^",
        color=val_boundary_color,
        linewidth=2.0,
        label="val_dice_boundary",
    )
    ax.plot(
        x,
        df["train_dice_interior"],
        linestyle=":",
        marker="D",
        color=train_interior_color,
        linewidth=2.0,
        alpha=0.85,
        label="train_dice_interior",
    )
    ax.plot(
        x,
        df["val_dice_interior"],
        linestyle="-",
        marker="D",
        color=val_interior_color,
        linewidth=2.0,
        label="val_dice_interior",
    )
    _apply_x_axis_style(ax, x, hide_x_axis)
    ax.set_ylabel("dice")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Mask Reconstruction: Boundary/Interior Dice vs Partion")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mask-reconstruction summary by partion/portion from synth_no_saca_mask_reconstruction.csv. "
            "Generates 3 figures: loss, dice, and boundary/interior dice."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to summary CSV.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: same folder as --input-csv).",
    )
    parser.add_argument("--train-loss-color", type=str, default="1f77b4", help="Hex color for train_loss_total.")
    parser.add_argument("--val-loss-color", type=str, default="d62728", help="Hex color for val_loss_total.")
    parser.add_argument("--train-dice-color", type=str, default="1f77b4", help="Hex color for train_dice.")
    parser.add_argument("--val-dice-color", type=str, default="d62728", help="Hex color for val_dice.")
    parser.add_argument(
        "--train-boundary-color",
        type=str,
        default="2ca02c",
        help="Hex color for train_dice_boundary.",
    )
    parser.add_argument(
        "--val-boundary-color",
        type=str,
        default="17becf",
        help="Hex color for val_dice_boundary.",
    )
    parser.add_argument(
        "--train-interior-color",
        type=str,
        default="ff7f0e",
        help="Hex color for train_dice_interior.",
    )
    parser.add_argument(
        "--val-interior-color",
        type=str,
        default="bcbd22",
        help="Hex color for val_dice_interior.",
    )
    parser.add_argument("--dpi-loss", type=int, default=170, help="DPI for loss figure.")
    parser.add_argument("--dpi-dice", type=int, default=170, help="DPI for main dice figure.")
    parser.add_argument("--dpi-dice-types", type=int, default=170, help="DPI for boundary/interior dice figure.")
    parser.add_argument(
        "--no-x-axis",
        action="store_true",
        help="Hide x-axis label, ticks, and bottom spine in all figures.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    out_dir = input_csv.parent if args.out_dir is None else Path(args.out_dir).expanduser().resolve()

    train_loss_color = _normalize_hex_color(args.train_loss_color)
    val_loss_color = _normalize_hex_color(args.val_loss_color)
    train_dice_color = _normalize_hex_color(args.train_dice_color)
    val_dice_color = _normalize_hex_color(args.val_dice_color)
    train_boundary_color = _normalize_hex_color(args.train_boundary_color)
    val_boundary_color = _normalize_hex_color(args.val_boundary_color)
    train_interior_color = _normalize_hex_color(args.train_interior_color)
    val_interior_color = _normalize_hex_color(args.val_interior_color)

    df = _load_csv(input_csv)
    hide_x_axis = bool(args.no_x_axis)

    out_loss = out_dir / "mask_recon_loss_vs_partion.svg"
    out_dice = out_dir / "mask_recon_dice_vs_partion.svg"
    out_dice_types = out_dir / "mask_recon_dice_types_vs_partion.svg"

    plot_loss_vs_partion(
        df=df,
        out_path=out_loss,
        train_loss_color=train_loss_color,
        val_loss_color=val_loss_color,
        dpi=int(args.dpi_loss),
        hide_x_axis=hide_x_axis,
    )
    plot_main_dice_vs_partion(
        df=df,
        out_path=out_dice,
        train_dice_color=train_dice_color,
        val_dice_color=val_dice_color,
        dpi=int(args.dpi_dice),
        hide_x_axis=hide_x_axis,
    )
    plot_other_dice_vs_partion(
        df=df,
        out_path=out_dice_types,
        train_boundary_color=train_boundary_color,
        val_boundary_color=val_boundary_color,
        train_interior_color=train_interior_color,
        val_interior_color=val_interior_color,
        dpi=int(args.dpi_dice_types),
        hide_x_axis=hide_x_axis,
    )

    print(f"[ok] Saved: {out_loss}")
    print(f"[ok] Saved: {out_dice}")
    print(f"[ok] Saved: {out_dice_types}")


if __name__ == "__main__":
    main()
