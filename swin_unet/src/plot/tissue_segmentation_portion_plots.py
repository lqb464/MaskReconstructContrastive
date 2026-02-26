from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _normalize_hex_color(color: str) -> str:
    value = str(color).strip()
    if not value.startswith("#"):
        value = f"#{value}"
    if len(value) != 7:
        raise ValueError(f"Invalid color '{color}'. Expected RRGGBB or #RRGGBB.")
    return value


def _load_summary_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"portion", "train_loss", "train_macro_dice", "val_loss", "val_macro_dice"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Invalid schema in {csv_path}. Missing columns: {sorted(missing)}")

    numeric_cols = ["portion", "train_loss", "train_macro_dice", "val_loss", "val_macro_dice"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols).copy()
    if df.empty:
        raise ValueError(f"No valid numeric rows in {csv_path}")

    return df.sort_values("portion", ascending=True).reset_index(drop=True)


def _class_id_from_col(col: str, prefix: str) -> int:
    tail = col.replace(prefix, "", 1)
    return int(tail.replace("_dice", ""))


def _collect_class_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    train_cols = sorted(
        [c for c in df.columns if c.startswith("train_class_") and c.endswith("_dice")],
        key=lambda c: _class_id_from_col(c, "train_class_"),
    )
    val_cols = sorted(
        [c for c in df.columns if c.startswith("val_class_") and c.endswith("_dice")],
        key=lambda c: _class_id_from_col(c, "val_class_"),
    )
    if not train_cols or not val_cols:
        raise ValueError("Class-dice columns not found. Expected train_class_*_dice and val_class_*_dice.")
    return train_cols, val_cols


def plot_loss_by_portion(
    df: pd.DataFrame,
    out_path: Path,
    train_color: str,
    val_color: str,
    dpi: int,
    hide_x_axis: bool,
) -> None:
    x = df["portion"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(x, df["train_loss"], linestyle="--", marker="o", color=train_color, linewidth=2.0, label="train_loss")
    ax.plot(x, df["val_loss"], linestyle="-", marker="o", color=val_color, linewidth=2.0, label="val_loss")

    if hide_x_axis:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xlabel("portion")
    ax.set_ylabel("loss")
    ax.set_title("Tissue Segmentation: Loss vs Portion")
    if not hide_x_axis:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x])
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=False)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def plot_macro_dice_by_portion(
    df: pd.DataFrame,
    out_path: Path,
    train_color: str,
    val_color: str,
    dpi: int,
    hide_x_axis: bool,
) -> None:
    x = df["portion"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(
        x,
        df["train_macro_dice"],
        linestyle="--",
        marker="s",
        color=train_color,
        linewidth=2.0,
        alpha=0.9,
        label="train_macro_dice",
    )
    ax.plot(
        x,
        df["val_macro_dice"],
        linestyle="-",
        marker="s",
        color=val_color,
        linewidth=2.0,
        alpha=0.9,
        label="val_macro_dice",
    )

    if hide_x_axis:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xlabel("portion")
    ax.set_ylabel("macro dice")
    ax.set_title("Tissue Segmentation: Macro Dice vs Portion")
    if not hide_x_axis:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x])
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=False)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def plot_class_dice_by_portion(
    df: pd.DataFrame,
    out_path: Path,
    cmap_name: str,
    train_alpha: float,
    val_alpha: float,
    linewidth: float,
    dpi: int,
    hide_x_axis: bool,
) -> None:
    train_cols, val_cols = _collect_class_columns(df)
    x = df["portion"].to_numpy(dtype=float)
    cmap = plt.get_cmap(cmap_name)

    n = min(len(train_cols), len(val_cols))
    fig, ax = plt.subplots(figsize=(14, 8))

    for i in range(n):
        train_col = train_cols[i]
        val_col = val_cols[i]
        class_id = _class_id_from_col(train_col, "train_class_")
        color = cmap(i % cmap.N)
        y_train = pd.to_numeric(df[train_col], errors="coerce").to_numpy(dtype=float)
        y_val = pd.to_numeric(df[val_col], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y_train).all() and np.isnan(y_val).all():
            continue
        ax.plot(
            x,
            y_train,
            linestyle="--",
            color=color,
            linewidth=float(linewidth),
            alpha=float(train_alpha),
            label=f"class_{class_id} train",
        )
        ax.plot(
            x,
            y_val,
            linestyle="-",
            color=color,
            linewidth=float(linewidth),
            alpha=float(val_alpha),
            label=f"class_{class_id} val",
        )

    if hide_x_axis:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xlabel("portion")
    ax.set_ylabel("dice")
    ax.set_title("Tissue Segmentation: Class Dice vs Portion")
    if not hide_x_axis:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x])
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize=7, frameon=False)
    plt.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot tissue-segmentation summary by data portion from partition_best_epoch_summary.csv. "
            "Generates 3 figures: loss_vs_portion, macro_dice_vs_portion, and class_dice_vs_portion."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to partition_best_epoch_summary.csv")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: same folder as --input-csv)",
    )
    parser.add_argument("--train-color", type=str, default="1f77b4", help="Hex color for train curves.")
    parser.add_argument("--val-color", type=str, default="d62728", help="Hex color for val curves.")
    parser.add_argument(
        "--train-macro-color",
        type=str,
        default=None,
        help="Hex color for train macro-dice curve (default: same as --train-color).",
    )
    parser.add_argument(
        "--val-macro-color",
        type=str,
        default=None,
        help="Hex color for val macro-dice curve (default: same as --val-color).",
    )
    parser.add_argument("--class-cmap", type=str, default="tab20", help="Colormap for class curves.")
    parser.add_argument("--class-linewidth", type=float, default=1.4, help="Line width for class-dice curves.")
    parser.add_argument("--class-train-alpha", type=float, default=0.55, help="Alpha for train class curves.")
    parser.add_argument("--class-val-alpha", type=float, default=0.9, help="Alpha for val class curves.")
    parser.add_argument("--dpi-loss", type=int, default=170, help="DPI for loss figure.")
    parser.add_argument("--dpi-macro", type=int, default=170, help="DPI for macro-dice figure.")
    parser.add_argument("--dpi-class", type=int, default=170, help="DPI for class-dice figure.")
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
    train_color = _normalize_hex_color(args.train_color)
    val_color = _normalize_hex_color(args.val_color)
    train_macro_color = train_color if args.train_macro_color is None else _normalize_hex_color(args.train_macro_color)
    val_macro_color = val_color if args.val_macro_color is None else _normalize_hex_color(args.val_macro_color)

    df = _load_summary_csv(input_csv)

    out_loss = out_dir / "tissue_loss_vs_portion.svg"
    out_macro = out_dir / "tissue_macro_dice_vs_portion.svg"
    out_class = out_dir / "tissue_class_dice_vs_portion.svg"

    plot_loss_by_portion(
        df=df,
        out_path=out_loss,
        train_color=train_color,
        val_color=val_color,
        dpi=int(args.dpi_loss),
        hide_x_axis=bool(args.no_x_axis),
    )
    plot_macro_dice_by_portion(
        df=df,
        out_path=out_macro,
        train_color=train_macro_color,
        val_color=val_macro_color,
        dpi=int(args.dpi_macro),
        hide_x_axis=bool(args.no_x_axis),
    )
    plot_class_dice_by_portion(
        df=df,
        out_path=out_class,
        cmap_name=str(args.class_cmap),
        train_alpha=float(args.class_train_alpha),
        val_alpha=float(args.class_val_alpha),
        linewidth=float(args.class_linewidth),
        dpi=int(args.dpi_class),
        hide_x_axis=bool(args.no_x_axis),
    )

    print(f"[ok] Saved: {out_loss}")
    print(f"[ok] Saved: {out_macro}")
    print(f"[ok] Saved: {out_class}")


if __name__ == "__main__":
    main()
