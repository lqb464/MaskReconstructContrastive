from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def _plot_primary_metric(df: pd.DataFrame, plot_dir: Path) -> None:

    if {"train_primary_metric", "eval_primary_metric"}.issubset(set(df.columns)):
        metric_name = "primary_metric"
        if "primary_metric_name" in df.columns:
            names = [str(v) for v in df["primary_metric_name"].dropna().unique().tolist() if str(v)]
            if names:
                metric_name = names[-1]
        _plot_two_lines(
            df,
            "epoch",
            "train_primary_metric",
            "eval_primary_metric",
            plot_dir / "primary_metric.png",
            metric_name,
        )
        return

    if {"train_pc_macro_dice", "eval_pc_macro_dice"}.issubset(set(df.columns)):
        _plot_two_lines(
            df,
            "epoch",
            "train_pc_macro_dice",
            "eval_pc_macro_dice",
            plot_dir / "primary_metric.png",
            "pc_macro_dice",
        )
        return

    if {"train_macro_dice", "eval_macro_dice"}.issubset(set(df.columns)):
        _plot_two_lines(
            df,
            "epoch",
            "train_macro_dice",
            "eval_macro_dice",
            plot_dir / "primary_metric.png",
            "macro_dice",
        )

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
        if class_name_map and cid not in class_name_map:

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

def _plot_per_class_dice_heatmap(
    df_long: pd.DataFrame,
    *,
    split: str,
    out_path: Path,
    class_name_map: dict[int, str],
) -> None:
    required = {"epoch", "split", "class_id", "dice"}
    if not required.issubset(set(df_long.columns)):
        return

    df = df_long[df_long["split"] == split].copy()
    if df.empty:
        return

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["class_id"] = pd.to_numeric(df["class_id"], errors="coerce")
    df["dice"] = pd.to_numeric(df["dice"], errors="coerce")
    df = df.dropna(subset=["epoch", "class_id", "dice"])
    if df.empty:
        return

    pivot = (
        df.pivot_table(index="class_id", columns="epoch", values="dice", aggfunc="mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    if pivot.empty:
        return

    data = pivot.values
    fig_h = max(6, min(20, 0.25 * data.shape[0] + 2))
    fig_w = max(8, min(24, 0.35 * data.shape[1] + 4))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("dice", rotation=90)

    epoch_vals = list(pivot.columns.astype(int))
    class_ids = list(pivot.index.astype(int))
    y_labels = [f"{cid}: {class_name_map.get(cid, f'class_{cid}')}" for cid in class_ids]

    x_step = max(1, len(epoch_vals) // 20)
    y_step = max(1, len(class_ids) // 40)
    ax.set_xticks(np.arange(0, len(epoch_vals), x_step))
    ax.set_xticklabels([str(epoch_vals[i]) for i in range(0, len(epoch_vals), x_step)], rotation=45, ha="right")
    ax.set_yticks(np.arange(0, len(class_ids), y_step))
    ax.set_yticklabels([y_labels[i] for i in range(0, len(class_ids), y_step)], fontsize=7)

    ax.set_xlabel("epoch")
    ax.set_ylabel("class")
    ax.set_title(f"Per-class {split.capitalize()} Dice Heatmap (All Labels)")
    ax.grid(False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()

def _plot_per_label_dice_heatmap_ranked(
    df_labels: pd.DataFrame,
    *,
    split: str,
    out_path: Path,
) -> None:
    required = {"epoch", "split", "enc_id", "label_name", "dice"}
    if not required.issubset(set(df_labels.columns)):
        return

    df = df_labels[df_labels["split"] == split].copy()
    if df.empty:
        return

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["enc_id"] = pd.to_numeric(df["enc_id"], errors="coerce")
    df["dice"] = pd.to_numeric(df["dice"], errors="coerce")
    df = df.dropna(subset=["epoch", "enc_id", "dice"])
    if df.empty:
        return

    name_map: dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            cid = int(row["enc_id"])
        except Exception:
            continue
        if cid not in name_map:
            name_map[cid] = str(row.get("label_name", f"class_{cid}"))

    pivot = (
        df.pivot_table(index="enc_id", columns="epoch", values="dice", aggfunc="mean")
        .sort_index(axis=1)
    )
    if pivot.empty:
        return

    latest_scores = pivot.iloc[:, -1].fillna(-1.0)
    ranked_ids = list(latest_scores.sort_values(ascending=False).index.astype(int))
    pivot = pivot.loc[ranked_ids]

    data = pivot.values
    fig_h = max(6, min(24, 0.28 * data.shape[0] + 2))
    fig_w = max(9, min(24, 0.35 * data.shape[1] + 4))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="#ececec")
    im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("dice", rotation=90)

    epoch_vals = list(pivot.columns.astype(int))
    y_labels = [f"{cid}: {name_map.get(cid, f'class_{cid}')}" for cid in ranked_ids]

    x_step = max(1, len(epoch_vals) // 20)
    y_step = max(1, len(ranked_ids) // 40)
    ax.set_xticks(np.arange(0, len(epoch_vals), x_step))
    ax.set_xticklabels([str(epoch_vals[i]) for i in range(0, len(epoch_vals), x_step)], rotation=45, ha="right")
    ax.set_yticks(np.arange(0, len(ranked_ids), y_step))
    ax.set_yticklabels([y_labels[i] for i in range(0, len(ranked_ids), y_step)], fontsize=7)

    ax.set_xlabel("epoch")
    ax.set_ylabel("label (ranked by latest dice)")
    ax.set_title(f"Per-label {split.capitalize()} Dice (Ranked Heatmap)")
    ax.grid(False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()

def _plot_per_label_latest_ranking(
    df_labels: pd.DataFrame,
    *,
    split: str,
    out_path: Path,
) -> None:
    required = {"epoch", "split", "enc_id", "label_name", "dice"}
    if not required.issubset(set(df_labels.columns)):
        return

    df = df_labels[df_labels["split"] == split].copy()
    if df.empty:
        return

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["enc_id"] = pd.to_numeric(df["enc_id"], errors="coerce")
    df["dice"] = pd.to_numeric(df["dice"], errors="coerce")
    df = df.dropna(subset=["epoch", "enc_id", "dice"])
    if df.empty:
        return

    latest_epoch = int(df["epoch"].max())
    latest = df[df["epoch"] == latest_epoch].copy()
    if latest.empty:
        return

    latest = latest.sort_values("dice", ascending=False)
    y_labels = [f"{int(r.enc_id)}: {str(r.label_name)}" for r in latest.itertuples()]
    y_pos = np.arange(len(latest))
    x_vals = latest["dice"].to_numpy(dtype=float)

    fig_h = max(6, min(24, 0.28 * len(latest) + 2))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.hlines(y=y_pos, xmin=0.0, xmax=x_vals, color="#4c78a8", linewidth=1.5, alpha=0.9)
    ax.plot(x_vals, y_pos, "o", color="#f58518", markersize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("dice")
    ax.set_ylabel("label")
    ax.set_title(f"Latest {split.capitalize()} Dice Ranking (epoch={latest_epoch})")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
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
    _plot_primary_metric(df, plot_dir)
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

    per_class_long_csv = csv_path.parent / "per_class_dice_by_split.csv"
    if per_class_long_csv.exists():
        df_long = pd.read_csv(per_class_long_csv)
        class_name_map = _read_class_name_map(csv_path.parent / "reports" / "class_id_to_name.json")
        _plot_per_class_dice_heatmap(
            df_long,
            split="eval",
            out_path=plot_dir / "per_class_eval_dice_heatmap_all_labels.png",
            class_name_map=class_name_map,
        )
        _plot_per_class_dice_heatmap(
            df_long,
            split="train",
            out_path=plot_dir / "per_class_train_dice_heatmap_all_labels.png",
            class_name_map=class_name_map,
        )

    per_label_csv = csv_path.parent / "per_label_dice.csv"
    if per_label_csv.exists():
        df_labels = pd.read_csv(per_label_csv)
        _plot_per_label_dice_heatmap_ranked(
            df_labels,
            split="eval",
            out_path=plot_dir / "per_label_eval_dice_ranked_heatmap.png",
        )
        _plot_per_label_dice_heatmap_ranked(
            df_labels,
            split="train",
            out_path=plot_dir / "per_label_train_dice_ranked_heatmap.png",
        )
        _plot_per_label_latest_ranking(
            df_labels,
            split="eval",
            out_path=plot_dir / "per_label_eval_dice_latest_ranking.png",
        )
        _plot_per_label_latest_ranking(
            df_labels,
            split="train",
            out_path=plot_dir / "per_label_train_dice_latest_ranking.png",
        )

__all__ = ["generate_plots"]
