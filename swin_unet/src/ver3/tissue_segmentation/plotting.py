from __future__ import annotations

import json
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


def _plot_per_class_eval_dice(
    df: pd.DataFrame,
    out_path: Path,
    class_name_map: dict[int, str],
) -> None:
    class_cols = [c for c in df.columns if c.startswith("class_")]
    if "epoch" not in df.columns or not class_cols:
        return

    plt.figure(figsize=(14, 8))
    plotted = 0
    for col in class_cols:
        try:
            cid = int(col.split("_", 1)[1])
        except Exception:
            continue
        y = pd.to_numeric(df[col], errors="coerce")
        if y.notna().sum() == 0:
            continue
        label = f"{cid}: {class_name_map.get(cid, f'class_{cid}')}"
        plt.plot(df["epoch"], y, linewidth=1.0, alpha=0.9, label=label)
        plotted += 1

    if plotted == 0:
        plt.close()
        return

    plt.xlabel("epoch")
    plt.ylabel("eval dice")
    plt.title("Per-class Eval Dice (All Labels)")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=6,
        ncol=1,
        frameon=False,
    )
    plt.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _read_class_name_map(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[int, str] = {}
    if isinstance(payload, dict):
        for k, v in payload.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
    return out


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

    per_class_csv = csv_path.parent / "per_class_dice_eval.csv"
    if per_class_csv.exists():
        per_class_df = pd.read_csv(per_class_csv)
        class_name_map = _read_class_name_map(csv_path.parent / "reports" / "class_id_to_name.json")
        _plot_per_class_eval_dice(
            per_class_df,
            plot_dir / "per_class_eval_dice_all_labels.png",
            class_name_map,
        )


__all__ = ["generate_plots"]
