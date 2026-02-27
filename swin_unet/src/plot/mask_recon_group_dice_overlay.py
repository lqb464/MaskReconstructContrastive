from __future__ import annotations

import argparse
import csv
import heapq
import json
from collections import Counter
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

import matplotlib
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from swin_unet.src.ver3.config.experiment import ExperimentConfig
from swin_unet.src.ver3.mask_reconstruction.dataset import MaskReconstructionDataset
from swin_unet.src.ver3.mask_reconstruction.main import build_model
from swin_unet.src.ver3.training.utils import ensure_dir, get_device

VALID_MODALITIES = ("t1", "ct", "pet", "t2", "dwi", "flair")


def dataclass_from_dict(dc_type, raw: dict):
    if not is_dataclass(dc_type):
        raise TypeError(f"{dc_type} is not a dataclass")

    type_hints = get_type_hints(dc_type)
    kwargs = {}
    for f in fields(dc_type):
        name = f.name
        if name not in raw:
            continue
        val = raw[name]
        ftype = type_hints.get(name, f.type)

        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[name] = dataclass_from_dict(ftype, val)
            continue

        origin = get_origin(ftype)
        args = get_args(ftype)
        if origin is Union and isinstance(val, dict):
            dc_candidates = [a for a in args if is_dataclass(a)]
            if dc_candidates:
                kwargs[name] = dataclass_from_dict(dc_candidates[0], val)
                continue

        kwargs[name] = val

    return dc_type(**kwargs)


def extract_group_from_name(path: str | Path) -> Optional[str]:
    token = extract_group_token_from_name(path)
    if token is None:
        return None
    if token not in VALID_MODALITIES:
        return None
    return token


def extract_group_token_from_name(path: str | Path) -> Optional[str]:
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    return parts[1].lower()


def per_sample_dice(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    denom = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3)) + eps
    return (2.0 * inter + eps) / denom


def _maybe_draw_contour(ax: plt.Axes, mask: np.ndarray, *, color: str, linewidth: float) -> None:
    if np.any(mask > 0):
        ax.contour(mask.astype(np.float32), levels=[0.5], colors=color, linewidths=linewidth)


def save_group_overlay(group: str, items: list[dict[str, Any]], out_path: Path) -> None:
    if not items:
        return

    items_sorted = sorted(items, key=lambda r: r["dice"], reverse=True)
    n = len(items_sorted)

    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n))
    if n == 1:
        axes = [axes]

    for ax, row in zip(axes, items_sorted):
        image = row["image"]
        target = row["target"]
        pred = row["pred"]
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        # Light GT mask fill to make GT presence visually obvious even when boundary is thin.
        if np.any(target > 0):
            ax.imshow(np.ma.masked_where(target <= 0, target), cmap="spring", alpha=0.18, vmin=0, vmax=1)
        _maybe_draw_contour(ax, target, color="lime", linewidth=1.4)
        _maybe_draw_contour(ax, pred, color="red", linewidth=1.2)
        ax.set_title(f"{Path(row['path']).name} | dice={row['dice']:.4f}", fontsize=10)
        ax.axis("off")

    handles = [
        Line2D([0], [0], color="lime", lw=2, label="GT boundary"),
        Line2D([0], [0], color="red", lw=2, label="Pred boundary"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False)
    fig.suptitle(f"Top Dice Overlays - group={group}", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def load_model_from_ckpt(
    *,
    ckpt_path: Path,
    device: torch.device,
    ckpt_load_mode: str,
    image_size_override: int,
) -> tuple[torch.nn.Module, ExperimentConfig]:
    if ckpt_load_mode != "full":
        raise ValueError("This script currently supports only --ckpt-load-mode full.")

    obj = torch.load(ckpt_path, map_location=device)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid checkpoint format: {ckpt_path}")
    if "cfg" not in obj:
        raise ValueError(f"Checkpoint missing cfg dict: {ckpt_path}")

    raw_cfg = obj["cfg"]
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Invalid cfg in checkpoint: {ckpt_path}")
    cfg = dataclass_from_dict(ExperimentConfig, raw_cfg)

    if int(image_size_override) > 0:
        cfg.data.image_size = int(image_size_override)
        cfg.mask.image_size = int(image_size_override)
    cfg.training.enable_contrastive = False

    model = build_model(cfg).to(device)

    state_dict = obj.get("model", None)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint missing model state_dict: {ckpt_path}")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, cfg


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate mask reconstruction checkpoint on a folder dataset, group by modality token in filename "
            "(*_x_*.png), compute dice stats per group, and save top-k boundary overlay visualizations."
        )
    )
    p.add_argument("--input-dir", type=str, required=True, help="Folder containing image/mask pairs.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    p.add_argument("--ckpt-load-mode", type=str, default="full", choices=["full"], help="Must be full.")
    p.add_argument("--out-dir", type=str, required=True, help="Output folder for csv/json and overlays.")

    p.add_argument("--image-ext", type=str, default=".png")
    p.add_argument("--mask-suffix", type=str, default="_mask.npz")
    p.add_argument("--mask-key", type=str, default="")
    p.add_argument("--strict-pairs", type=int, default=1, help="1: error on missing masks, 0: skip missing")

    p.add_argument("--image-size", type=int, default=0, help="Override checkpoint image size (0 keeps ckpt cfg)")
    p.add_argument("--target-size", type=int, default=0)
    p.add_argument("--resize-mode", type=str, default="letterbox", choices=["letterbox", "direct"])
    p.add_argument("--plane", type=str, default="axial", choices=["axial", "coronal", "auto"])
    p.add_argument(
        "--binarize-target",
        action="store_true",
        default=True,
        help="Binarize loaded target mask before evaluation. Default: enabled.",
    )
    p.add_argument(
        "--no-binarize-target",
        dest="binarize_target",
        action="store_false",
        help="Disable target binarization in dataset loader.",
    )
    p.add_argument(
        "--target-threshold",
        type=float,
        default=0.0,
        help="Threshold to convert target tensor to binary for Dice/overlay (default 0.0).",
    )
    p.add_argument("--threshold", type=float, default=0.5, help="Dice/pred threshold.")

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--top-k", type=int, default=5)
    return p


def run(args: argparse.Namespace) -> None:
    device = get_device(cpu=bool(args.cpu))
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    overlay_dir = ensure_dir(out_dir / "overlays")

    model, cfg = load_model_from_ckpt(
        ckpt_path=ckpt_path,
        device=device,
        ckpt_load_mode=args.ckpt_load_mode,
        image_size_override=int(args.image_size),
    )

    dataset = MaskReconstructionDataset(
        data_dir=args.input_dir,
        image_ext=args.image_ext,
        mask_suffix=args.mask_suffix,
        strict_pairs=bool(args.strict_pairs),
        mask_key=(args.mask_key or None),
        image_size=int(cfg.data.image_size) if int(cfg.data.image_size) > 0 else None,
        target_size=int(args.target_size),
        resize_mode=args.resize_mode,
        plane=args.plane,
        binarize_target=bool(args.binarize_target),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory) and device.type == "cuda",
        drop_last=False,
    )

    top_k = max(1, int(args.top_k))
    heap_by_group: dict[str, list[tuple[float, int, dict[str, Any]]]] = {g: [] for g in VALID_MODALITIES}
    heap_by_group_non_empty_gt: dict[str, list[tuple[float, int, dict[str, Any]]]] = {g: [] for g in VALID_MODALITIES}
    stats = {g: {"n": 0, "sum": 0.0, "sum_sq": 0.0} for g in VALID_MODALITIES}
    group_empty_stats = {g: {"gt_empty": 0, "pred_empty": 0, "both_empty": 0, "gt_non_empty": 0} for g in VALID_MODALITIES}
    all_prefix_groups = Counter()
    per_image_rows: list[dict[str, Any]] = []
    skipped = 0
    gt_empty = 0
    serial = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            plane = batch["plane_one_hot"].to(device, non_blocking=True)
            paths = batch["path"]

            recon1, _, _, _ = model(x, None, plane)
            pred = (torch.sigmoid(recon1) >= float(args.threshold)).float()
            # Robust target binarization: default threshold 0.0 works for both
            # binarized masks and masks stored as small scaled values (e.g., /255 path).
            tgt = (y > float(args.target_threshold)).float()
            dice_vals = per_sample_dice(pred, tgt)

            for i, path in enumerate(paths):
                token = extract_group_token_from_name(path)
                if token is not None:
                    all_prefix_groups[token] += 1

                group = extract_group_from_name(path)
                if group is None:
                    skipped += 1
                    continue

                dice_i = float(dice_vals[i].item())
                gt_pixels = float(tgt[i].sum().item())
                pred_pixels = float(pred[i].sum().item())
                gt_is_empty = gt_pixels <= 0.0
                pred_is_empty = pred_pixels <= 0.0

                gstat = group_empty_stats[group]
                if gt_is_empty:
                    gt_empty += 1
                    gstat["gt_empty"] += 1
                else:
                    gstat["gt_non_empty"] += 1
                if pred_is_empty:
                    gstat["pred_empty"] += 1
                if gt_is_empty and pred_is_empty:
                    gstat["both_empty"] += 1
                per_image_rows.append({"path": str(path), "group": group, "dice": dice_i})

                st = stats[group]
                st["n"] += 1
                st["sum"] += dice_i
                st["sum_sq"] += dice_i * dice_i

                payload = {
                    "path": str(path),
                    "dice": dice_i,
                    "image": x[i, 0].detach().cpu().numpy(),
                    "target": tgt[i, 0].detach().cpu().numpy(),
                    "pred": pred[i, 0].detach().cpu().numpy(),
                    "gt_pixels": gt_pixels,
                    "pred_pixels": pred_pixels,
                }

                serial += 1
                bucket = heap_by_group[group]
                entry = (dice_i, serial, payload)
                if len(bucket) < top_k:
                    heapq.heappush(bucket, entry)
                elif dice_i > bucket[0][0]:
                    heapq.heapreplace(bucket, entry)

                # For visualization quality: rank top-k using only samples with non-empty GT mask.
                if not gt_is_empty:
                    bucket_fg = heap_by_group_non_empty_gt[group]
                    if len(bucket_fg) < top_k:
                        heapq.heappush(bucket_fg, entry)
                    elif dice_i > bucket_fg[0][0]:
                        heapq.heapreplace(bucket_fg, entry)

    summary_rows: list[dict[str, Any]] = []
    for group in VALID_MODALITIES:
        st = stats[group]
        n = int(st["n"])
        if n == 0:
            mean = float("nan")
            std = float("nan")
        else:
            mean = float(st["sum"] / n)
            var = max(0.0, float(st["sum_sq"] / n - mean * mean))
            std = float(var ** 0.5)
        summary_rows.append({"group": group, "count": n, "avg_dice": mean, "std_dice": std})

    per_image_csv = out_dir / "per_image_dice.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "group", "dice"])
        writer.writeheader()
        writer.writerows(per_image_rows)

    summary_csv = out_dir / "group_dice_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "count", "avg_dice", "std_dice"])
        writer.writeheader()
        writer.writerows(summary_rows)

    for group in VALID_MODALITIES:
        top_items = [entry[2] for entry in heap_by_group_non_empty_gt[group]]
        if not top_items:
            # Fallback only when group has no non-empty GT at all.
            top_items = [entry[2] for entry in heap_by_group[group]]
        if not top_items:
            continue
        save_group_overlay(group, top_items, overlay_dir / f"{group}_top{top_k}_overlay.png")

    summary_json = {
        "checkpoint": str(ckpt_path),
        "input_dir": str(Path(args.input_dir).expanduser().resolve()),
        "threshold": float(args.threshold),
        "target_threshold": float(args.target_threshold),
        "top_k": top_k,
        "valid_groups": list(VALID_MODALITIES),
        "all_prefix_groups": dict(sorted(all_prefix_groups.items(), key=lambda kv: kv[0])),
        "skipped_images_without_valid_group_token": int(skipped),
        "gt_empty_after_threshold": int(gt_empty),
        "group_empty_stats": group_empty_stats,
        "summary": summary_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    print(f"[done] checkpoint={ckpt_path}")
    print(f"[done] evaluated_images={len(per_image_rows)} skipped={skipped}")
    print(f"[done] gt_empty_after_threshold={gt_empty} (target_threshold={float(args.target_threshold):.6f})")
    print(f"[done] per-image csv: {per_image_csv}")
    print(f"[done] summary csv: {summary_csv}")
    print(f"[done] overlay dir: {overlay_dir}")
    if all_prefix_groups:
        print("[groups] all prefix groups (token after first '_'):")
        for g, n in sorted(all_prefix_groups.items(), key=lambda kv: kv[0]):
            print(f"[groups] {g}: {n}")
    else:
        print("[groups] no valid filename token found after first '_'")
    for row in summary_rows:
        print(
            f"[group] {row['group']}: n={row['count']} "
            f"avg_dice={row['avg_dice']:.4f} std_dice={row['std_dice']:.4f}"
        )
    for g in VALID_MODALITIES:
        gs = group_empty_stats[g]
        print(
            f"[group-empty] {g}: gt_non_empty={gs['gt_non_empty']} gt_empty={gs['gt_empty']} "
            f"pred_empty={gs['pred_empty']} both_empty={gs['both_empty']}"
        )


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
