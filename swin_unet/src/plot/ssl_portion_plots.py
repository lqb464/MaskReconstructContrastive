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


def _load_ssl_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"portion", "train_loss", "train_ssim", "val_loss", "val_ssim"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Invalid schema in {csv_path}. Missing columns: {sorted(missing)}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required)).copy()
    if df.empty:
        raise ValueError(f"No valid numeric rows in {csv_path}")

    return df.sort_values("portion", ascending=True).reset_index(drop=True)


def _apply_x_axis_style(ax: plt.Axes, x_vals, hide_x_axis: bool) -> None:
    if hide_x_axis:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xlabel("portion")
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{v:g}" for v in x_vals])


def _plot_metric(
    *,
    df: pd.DataFrame,
    y_train_col: str,
    y_val_col: str,
    out_path: Path,
    train_color: str,
    val_color: str,
    y_label: str,
    title: str,
    dpi: int,
    hide_x_axis: bool,
    y_lim: tuple[float, float] | None = None,
) -> None:
    x = df["portion"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(
        x,
        df[y_train_col],
        linestyle=":",
        marker="o",
        color=train_color,
        linewidth=2.0,
        label=y_train_col,
    )
    ax.plot(
        x,
        df[y_val_col],
        linestyle="-",
        marker="o",
        color=val_color,
        linewidth=2.0,
        label=y_val_col,
    )

    _apply_x_axis_style(ax, x, hide_x_axis)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot SSL summary by portion from 2 CSVs: ixi_multi_saca_ssl.csv and ixi_no_saca_ssl.csv. "
            "Generates 4 figures: 2 for loss and 2 for SSIM."
        )
    )
    parser.add_argument("--multi-csv", type=Path, required=True, help="Path to ixi_multi_saca_ssl.csv")
    parser.add_argument("--no-saca-csv", type=Path, required=True, help="Path to ixi_no_saca_ssl.csv")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: same folder as --multi-csv)",
    )

    parser.add_argument("--multi-loss-train-color", type=str, default="1f77b4", help="Color for multi train_loss.")
    parser.add_argument("--multi-loss-val-color", type=str, default="d62728", help="Color for multi val_loss.")
    parser.add_argument("--multi-ssim-train-color", type=str, default="2ca02c", help="Color for multi train_ssim.")
    parser.add_argument("--multi-ssim-val-color", type=str, default="ff7f0e", help="Color for multi val_ssim.")

    parser.add_argument("--no-loss-train-color", type=str, default="9467bd", help="Color for no_saca train_loss.")
    parser.add_argument("--no-loss-val-color", type=str, default="8c564b", help="Color for no_saca val_loss.")
    parser.add_argument("--no-ssim-train-color", type=str, default="17becf", help="Color for no_saca train_ssim.")
    parser.add_argument("--no-ssim-val-color", type=str, default="bcbd22", help="Color for no_saca val_ssim.")

    parser.add_argument("--dpi-loss", type=int, default=170, help="DPI for both loss figures.")
    parser.add_argument("--dpi-ssim", type=int, default=170, help="DPI for both SSIM figures.")
    parser.add_argument(
        "--no-x-axis",
        action="store_true",
        help="Hide x-axis label, ticks, and bottom spine in all figures.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    multi_csv = Path(args.multi_csv).expanduser().resolve()
    no_csv = Path(args.no_saca_csv).expanduser().resolve()
    out_dir = multi_csv.parent if args.out_dir is None else Path(args.out_dir).expanduser().resolve()
    hide_x_axis = bool(args.no_x_axis)

    multi_df = _load_ssl_csv(multi_csv)
    no_df = _load_ssl_csv(no_csv)

    multi_loss_train_color = _normalize_hex_color(args.multi_loss_train_color)
    multi_loss_val_color = _normalize_hex_color(args.multi_loss_val_color)
    multi_ssim_train_color = _normalize_hex_color(args.multi_ssim_train_color)
    multi_ssim_val_color = _normalize_hex_color(args.multi_ssim_val_color)
    no_loss_train_color = _normalize_hex_color(args.no_loss_train_color)
    no_loss_val_color = _normalize_hex_color(args.no_loss_val_color)
    no_ssim_train_color = _normalize_hex_color(args.no_ssim_train_color)
    no_ssim_val_color = _normalize_hex_color(args.no_ssim_val_color)

    out_multi_loss = out_dir / "ssl_multi_saca_loss_vs_portion.svg"
    out_multi_ssim = out_dir / "ssl_multi_saca_ssim_vs_portion.svg"
    out_no_loss = out_dir / "ssl_no_saca_loss_vs_portion.svg"
    out_no_ssim = out_dir / "ssl_no_saca_ssim_vs_portion.svg"

    _plot_metric(
        df=multi_df,
        y_train_col="train_loss",
        y_val_col="val_loss",
        out_path=out_multi_loss,
        train_color=multi_loss_train_color,
        val_color=multi_loss_val_color,
        y_label="loss",
        title="SSL Multi-SACA: Loss vs Portion",
        dpi=int(args.dpi_loss),
        hide_x_axis=hide_x_axis,
    )
    _plot_metric(
        df=multi_df,
        y_train_col="train_ssim",
        y_val_col="val_ssim",
        out_path=out_multi_ssim,
        train_color=multi_ssim_train_color,
        val_color=multi_ssim_val_color,
        y_label="ssim",
        title="SSL Multi-SACA: SSIM vs Portion",
        dpi=int(args.dpi_ssim),
        hide_x_axis=hide_x_axis,
        y_lim=(0.0, 1.0),
    )
    _plot_metric(
        df=no_df,
        y_train_col="train_loss",
        y_val_col="val_loss",
        out_path=out_no_loss,
        train_color=no_loss_train_color,
        val_color=no_loss_val_color,
        y_label="loss",
        title="SSL No-SACA: Loss vs Portion",
        dpi=int(args.dpi_loss),
        hide_x_axis=hide_x_axis,
    )
    _plot_metric(
        df=no_df,
        y_train_col="train_ssim",
        y_val_col="val_ssim",
        out_path=out_no_ssim,
        train_color=no_ssim_train_color,
        val_color=no_ssim_val_color,
        y_label="ssim",
        title="SSL No-SACA: SSIM vs Portion",
        dpi=int(args.dpi_ssim),
        hide_x_axis=hide_x_axis,
        y_lim=(0.0, 1.0),
    )

    print(f"[ok] Saved: {out_multi_loss}")
    print(f"[ok] Saved: {out_multi_ssim}")
    print(f"[ok] Saved: {out_no_loss}")
    print(f"[ok] Saved: {out_no_ssim}")


if __name__ == "__main__":
    main()
