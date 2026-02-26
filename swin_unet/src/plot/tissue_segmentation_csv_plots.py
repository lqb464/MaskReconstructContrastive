from __future__ import annotations

import argparse
import json
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


def _read_class_name_map(run_dir: Path) -> dict[int, str]:
    report_map = run_dir / "reports" / "class_id_to_name.json"
    if report_map.exists():
        try:
            payload = json.loads(report_map.read_text(encoding="utf-8"))
            out: dict[int, str] = {}
            if isinstance(payload, dict):
                for k, v in payload.items():
                    out[int(k)] = str(v)
            if out:
                return out
        except Exception:
            pass

    per_label_csv = run_dir / "per_label_dice.csv"
    if not per_label_csv.exists():
        return {}
    try:
        df = pd.read_csv(per_label_csv)
    except Exception:
        return {}
    required = {"enc_id", "label_name"}
    if not required.issubset(set(df.columns)):
        return {}
    out = {}
    for row in df.itertuples():
        try:
            cid = int(row.enc_id)
            if cid not in out:
                out[cid] = str(row.label_name)
        except Exception:
            continue
    return out


def plot_per_class_eval_dice_all_labels(
    *,
    run_dir: Path,
    out_path: Path,
    class_name_map: dict[int, str],
    cmap_name: str,
    alpha: float,
    linewidth: float,
    legend_fontsize: int,
    dpi: int,
) -> None:
    csv_path = run_dir / "per_class_dice_eval.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)
    class_cols = [c for c in df.columns if c.startswith("class_")]
    if "epoch" not in df.columns or not class_cols:
        raise ValueError(f"Invalid schema in {csv_path}: expected 'epoch' and class_* columns.")

    cmap = plt.get_cmap(cmap_name)
    plt.figure(figsize=(14, 8))
    plotted = 0
    for i, col in enumerate(class_cols):
        try:
            cid = int(col.split("_", 1)[1])
        except Exception:
            continue
        y = pd.to_numeric(df[col], errors="coerce")
        if y.notna().sum() == 0:
            continue
        color = cmap(i % cmap.N)
        label = f"{cid}: {class_name_map.get(cid, f'class_{cid}')}"
        plt.plot(df["epoch"], y, linewidth=float(linewidth), alpha=float(alpha), color=color, label=label)
        plotted += 1

    if plotted == 0:
        plt.close()
        raise RuntimeError(f"No usable class curves found in {csv_path}.")

    plt.xlabel("epoch")
    plt.ylabel("eval dice")
    plt.title("Per-class Eval Dice (All Labels)")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=int(legend_fontsize),
        ncol=1,
        frameon=False,
    )
    plt.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def plot_per_label_eval_dice_latest_ranking(
    *,
    run_dir: Path,
    out_path: Path,
    line_color: str,
    point_color: str,
    dpi: int,
) -> None:
    csv_path = run_dir / "per_label_dice.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"epoch", "split", "enc_id", "label_name", "dice"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Invalid schema in {csv_path}: expected {sorted(required)}")

    df = df[df["split"] == "eval"].copy()
    if df.empty:
        raise RuntimeError(f"No eval rows found in {csv_path}.")

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["enc_id"] = pd.to_numeric(df["enc_id"], errors="coerce")
    df["dice"] = pd.to_numeric(df["dice"], errors="coerce")
    df = df.dropna(subset=["epoch", "enc_id", "dice"])
    if df.empty:
        raise RuntimeError(f"No valid numeric eval rows found in {csv_path}.")

    latest_epoch = int(df["epoch"].max())
    latest = df[df["epoch"] == latest_epoch].copy()
    if latest.empty:
        raise RuntimeError(f"No rows for latest eval epoch={latest_epoch} in {csv_path}.")

    latest = latest.sort_values("dice", ascending=False)
    y_labels = [f"{int(r.enc_id)}: {str(r.label_name)}" for r in latest.itertuples()]
    y_pos = np.arange(len(latest))
    x_vals = latest["dice"].to_numpy(dtype=float)

    fig_h = max(6, min(24, 0.28 * len(latest) + 2))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.hlines(y=y_pos, xmin=0.0, xmax=x_vals, color=line_color, linewidth=1.5, alpha=0.9)
    ax.plot(x_vals, y_pos, "o", color=point_color, markersize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("dice")
    ax.set_ylabel("label")
    ax.set_title(f"Latest Eval Dice Ranking (epoch={latest_epoch})")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild tissue-segmentation plots from CSV outputs. "
            "Creates: per_class_eval_dice_all_labels.png and per_label_eval_dice_latest_ranking.png."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Run folder containing CSV files.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder for PNG files (default: same as --input-dir).",
    )
    parser.add_argument(
        "--class-cmap",
        type=str,
        default="tab20",
        help="Matplotlib colormap name for per-class lines (default: tab20).",
    )
    parser.add_argument("--class-alpha", type=float, default=0.9, help="Line alpha for per-class plot.")
    parser.add_argument("--class-linewidth", type=float, default=1.0, help="Line width for per-class plot.")
    parser.add_argument("--legend-fontsize", type=int, default=6, help="Legend font size for per-class plot.")
    parser.add_argument(
        "--ranking-line-color",
        type=str,
        default="4c78a8",
        help="Hex color for horizontal lines in ranking plot.",
    )
    parser.add_argument(
        "--ranking-point-color",
        type=str,
        default="f58518",
        help="Hex color for points in ranking plot.",
    )
    parser.add_argument("--dpi-class", type=int, default=150, help="DPI for per_class_eval_dice_all_labels.png")
    parser.add_argument("--dpi-ranking", type=int, default=170, help="DPI for per_label_eval_dice_latest_ranking.png")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_dir = Path(args.input_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {run_dir}")
    out_dir = run_dir if args.out_dir is None else Path(args.out_dir).expanduser().resolve()

    ranking_line_color = _normalize_hex_color(args.ranking_line_color)
    ranking_point_color = _normalize_hex_color(args.ranking_point_color)
    class_name_map = _read_class_name_map(run_dir)

    out_class = out_dir / "per_class_eval_dice_all_labels.png"
    out_rank = out_dir / "per_label_eval_dice_latest_ranking.png"

    plot_per_class_eval_dice_all_labels(
        run_dir=run_dir,
        out_path=out_class,
        class_name_map=class_name_map,
        cmap_name=str(args.class_cmap),
        alpha=float(args.class_alpha),
        linewidth=float(args.class_linewidth),
        legend_fontsize=int(args.legend_fontsize),
        dpi=int(args.dpi_class),
    )
    plot_per_label_eval_dice_latest_ranking(
        run_dir=run_dir,
        out_path=out_rank,
        line_color=ranking_line_color,
        point_color=ranking_point_color,
        dpi=int(args.dpi_ranking),
    )

    print(f"[ok] Saved: {out_class}")
    print(f"[ok] Saved: {out_rank}")


if __name__ == "__main__":
    main()
