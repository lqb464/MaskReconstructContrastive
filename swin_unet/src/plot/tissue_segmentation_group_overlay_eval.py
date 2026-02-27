from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import re

from swin_unet.src.ver3.data.dataset import infer_plane_from_path
from swin_unet.src.ver3.data.dataset import plane_to_one_hot
from swin_unet.src.ver3.mask_reconstruction.pair_transforms import apply_pair_transforms
from swin_unet.src.ver3.mask_reconstruction.pair_transforms import load_image_pil
from swin_unet.src.ver3.models.model_utils import flip_lr
from swin_unet.src.ver3.models.swin_unet_dualview_ssl import SwinUNetDualViewSSL
from swin_unet.src.ver3.models.unet_dualview_ssl import UNetDualViewSSL
from swin_unet.src.ver3.tissue_segmentation.dataset import TissueSegmentationDataset
from swin_unet.src.ver3.tissue_segmentation.io import build_image_index
from swin_unet.src.ver3.tissue_segmentation.io import build_label_encoding_info
from swin_unet.src.ver3.tissue_segmentation.io import encode_label_array
from swin_unet.src.ver3.tissue_segmentation.io import identify_special_ids
from swin_unet.src.ver3.tissue_segmentation.io import load_label_array
from swin_unet.src.ver3.tissue_segmentation.io import parse_seg_labels_txt


VALID_GROUPS = ("t1", "ct", "pet", "t2", "dwi", "flair")


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path
    group: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate tissue segmentation checkpoint on an image folder, grouped by modality token in filename "
            "pattern '*_x_*.png' where x in {t1,ct,pet,t2,dwi,flair}. "
            "Computes Dice and exports boundary overlays for top-K images per group."
        )
    )
    p.add_argument("--input-dir", type=Path, required=True, help="Root folder containing PNG images.")
    p.add_argument(
        "--label-dir",
        type=Path,
        default=None,
        help="Root folder containing label files. Default: same as --input-dir.",
    )
    p.add_argument(
        "--label-suffix",
        type=str,
        default="_label.npz",
        help="Suffix for label files: <stem><label-suffix>.",
    )
    p.add_argument(
        "--label-suffixes",
        type=str,
        default="",
        help=(
            "Optional comma-separated suffix list for label lookup, e.g. "
            "'_label.npz,_label.npy,.npz,.npy'. If empty, script auto-adds common fallbacks."
        ),
    )
    p.add_argument("--label-key", type=str, default="", help="Optional key in NPZ label file.")
    p.add_argument("--seg-labels", type=Path, required=True, help="Path to seg_labels.txt.")
    p.add_argument("--mode", type=int, default=4, choices=[1, 2, 3, 4], help="Label encoding mode. Use 4 as requested.")
    p.add_argument(
        "--num-classes",
        type=int,
        default=0,
        help="Optional num_classes override for encoding info (0=infer).",
    )
    p.add_argument("--strict-label-ids", action="store_true", help="Require all label ids to exist in seg_labels.")
    p.add_argument("--no-strict-label-ids", dest="strict_label_ids", action="store_false")
    p.add_argument(
        "--allow-unknown-label-ids",
        action="store_true",
        help="Map unknown ids to class 0 when strict mode is off.",
    )
    p.add_argument("--no-allow-unknown-label-ids", dest="allow_unknown_label_ids", action="store_false")
    p.add_argument("--image-size", type=int, default=192, help="Square size for pair transform.")
    p.add_argument(
        "--resize-mode",
        type=str,
        default="letterbox",
        choices=["letterbox", "direct"],
        help="Resize strategy.",
    )
    p.add_argument("--ckpt-path", type=Path, required=True, help="Checkpoint path (.pt).")
    p.add_argument(
        "--ckpt-load-mode",
        type=str,
        default="full",
        choices=["full"],
        help="Checkpoint loading mode. This script uses full loading only.",
    )
    p.add_argument("--device", type=str, default="cuda", help="Device: cuda|cpu.")
    p.add_argument("--batch-size", type=int, default=8, help="Inference batch size.")
    p.add_argument("--top-k", type=int, default=5, help="Top-K overlays per group.")
    p.add_argument(
        "--group-by-modality",
        action="store_true",
        help="If enabled, group by modality token (t1/ct/pet/t2/dwi/flair). Default is disabled: all images go to group 'all'.",
    )
    p.add_argument("--include-bg", action="store_true", help="Include class 0 when computing macro dice.")
    p.add_argument(
        "--exclude-rank-label-ids",
        type=str,
        default="100,101,102,103,104,105",
        help="Comma-separated label ids. Samples whose target labels are only within this set are excluded from ranking.",
    )
    p.add_argument(
        "--rank-min-other-classes",
        type=int,
        default=1,
        help=(
            "Minimum number of target classes outside --exclude-rank-label-ids "
            "(after removing ignore ids {0,-100}) required to keep a sample in ranking."
        ),
    )
    p.add_argument("--overlay-dpi", type=int, default=220, help="DPI for overlay images.")
    p.add_argument("--overlay-linewidth", type=float, default=2.0, help="Boundary line width.")
    p.add_argument("--overlay-contrast-qmin", type=float, default=1.0, help="Lower percentile for image contrast.")
    p.add_argument("--overlay-contrast-qmax", type=float, default=99.0, help="Upper percentile for image contrast.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    p.set_defaults(
        strict_label_ids=False,
        allow_unknown_label_ids=True,
    )
    return p.parse_args()


def _mode4_guard(mode: int) -> None:
    if int(mode) != 4:
        raise ValueError(f"This run is required to use --mode 4, got mode={mode}.")


def _resolve_group(image_path: Path) -> str | None:
    """
    Resolve modality group from filename/parent names.
    Priority:
    1) Any underscore-separated token in filename stem
    2) Regex boundary-like match in filename stem
    3) Any token in parent directory names (nearest first)
    """
    stem = image_path.stem.lower()
    parts = [x for x in stem.split("_") if x]
    for tok in parts:
        if tok in VALID_GROUPS:
            return tok

    pat = re.compile(r"(?:^|[_\-.])(t1|ct|pet|t2|dwi|flair)(?:$|[_\-.])", flags=re.IGNORECASE)
    m = pat.search(stem)
    if m is not None:
        tok = str(m.group(1)).lower()
        if tok in VALID_GROUPS:
            return tok

    for parent in image_path.parents:
        name = parent.name.lower().strip()
        if not name:
            continue
        if name in VALID_GROUPS:
            return name
        pparts = [x for x in name.split("_") if x]
        for tok in pparts:
            if tok in VALID_GROUPS:
                return tok
        m2 = pat.search(name)
        if m2 is not None:
            tok = str(m2.group(1)).lower()
            if tok in VALID_GROUPS:
                return tok
    return None


def _build_label_index(label_root: Path, label_suffix: str) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for p in sorted(label_root.rglob("*")):
        if not p.is_file():
            continue
        if not p.name.endswith(label_suffix):
            continue
        stem = p.name[: -len(label_suffix)]
        out.setdefault(stem.lower(), []).append(p.resolve())
    return out


def _resolve_label_suffixes(primary_suffix: str, raw_suffixes: str) -> list[str]:
    suffixes: list[str] = []
    if raw_suffixes.strip():
        for x in raw_suffixes.split(","):
            sx = x.strip()
            if sx:
                suffixes.append(sx)

    if primary_suffix and primary_suffix not in suffixes:
        suffixes.insert(0, primary_suffix)

    # Common fallbacks in tissue segmentation pipelines.
    for sx in ("_label.npz", "_label.npy", ".npz", ".npy"):
        if sx not in suffixes:
            suffixes.append(sx)
    return suffixes


def _resolve_label_path(
    image_path: Path,
    image_root: Path,
    label_root: Path,
    label_suffixes: list[str],
    label_stem_indices: dict[str, dict[str, list[Path]]],
) -> Path | None:
    rel = image_path.resolve().relative_to(image_root.resolve())
    for sx in label_suffixes:
        c1 = (label_root / rel.parent / f"{image_path.stem}{sx}").resolve()
        if c1.exists():
            return c1

    for sx in label_suffixes:
        idx = label_stem_indices.get(sx, {})
        cands = idx.get(image_path.stem.lower(), [])
        if cands:
            return sorted(cands)[0]
    return None


def _collect_samples(
    dataset: TissueSegmentationDataset,
    *,
    group_by_modality: bool,
) -> list[Sample]:
    samples: list[Sample] = []
    total_png = int(dataset.num_images_resolved)
    dropped_group = 0
    dropped_label = int(dataset.num_missing_labels)
    grouped_counts: dict[str, int] = {}
    paired_counts: dict[str, int] = {}
    missing_label_examples: list[str] = []  # source dataset already handles detailed filtering; keep empty here.
    for (img_path, lbl_path) in dataset.pairs:
        img = Path(img_path)
        if bool(group_by_modality):
            group = _resolve_group(img)
            if group is None:
                continue
        else:
            group = "all"
        grouped_counts[group] = grouped_counts.get(group, 0) + 1
        samples.append(Sample(image_path=img.resolve(), label_path=Path(lbl_path).resolve(), group=group))
        paired_counts[group] = paired_counts.get(group, 0) + 1

    if bool(group_by_modality):
        for img in dataset.images:
            g = _resolve_group(Path(img))
            if g is None:
                dropped_group += 1
            else:
                grouped_counts[g] = grouped_counts.get(g, 0) + 1
    else:
        grouped_counts["all"] = total_png

    ordered_groups = sorted(set(grouped_counts.keys()) | set(paired_counts.keys()))
    grouped_breakdown = ", ".join(f"{g}:{grouped_counts.get(g, 0)}" for g in ordered_groups)
    paired_breakdown = ", ".join(f"{g}:{paired_counts.get(g, 0)}" for g in ordered_groups)
    print(
        f"[scan] total_png={total_png} grouped={total_png - dropped_group} "
        f"paired={len(samples)} dropped_group={dropped_group} dropped_label={dropped_label} "
        f"grouped_by_group={{ {grouped_breakdown} }} paired_by_group={{ {paired_breakdown} }}"
    )
    if missing_label_examples:
        print("[scan] missing_label_examples:")
        for p in missing_label_examples:
            print(f"  - {p}")
    return samples


def _to_device(device_arg: str) -> torch.device:
    d = str(device_arg).lower()
    if d == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_ckpt(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    obj = torch.load(path, map_location=device)
    if "model" not in obj:
        raise KeyError(f"Checkpoint missing 'model' key: {path}")
    return obj


def _get_cfg_value(cfg: dict[str, Any], section: str, key: str, default: Any) -> Any:
    s = cfg.get(section, {})
    if isinstance(s, dict):
        return s.get(key, default)
    return default


def _build_model_from_ckpt_cfg(
    ckpt_cfg: dict[str, Any],
    *,
    num_classes: int,
    image_size: int,
) -> torch.nn.Module:
    model_cfg = ckpt_cfg.get("model", {}) if isinstance(ckpt_cfg.get("model", {}), dict) else {}
    train_cfg = ckpt_cfg.get("training", {}) if isinstance(ckpt_cfg.get("training", {}), dict) else {}
    contrast_cfg = ckpt_cfg.get("contrast_loss", {}) if isinstance(ckpt_cfg.get("contrast_loss", {}), dict) else {}

    backbone = str(model_cfg.get("backbone", "swin")).lower()
    if backbone == "unet":
        return UNetDualViewSSL(
            in_ch=int(model_cfg.get("in_ch", 1)),
            base_ch=int(model_cfg.get("unet_base_ch", 16)),
            out_ch=int(num_classes),
            use_gn=bool(model_cfg.get("unet_use_gn", False)),
            use_se=bool(model_cfg.get("unet_use_se", False)),
            enable_reconstruct=bool(train_cfg.get("enable_reconstruct", True)),
            enable_contrastive=bool(train_cfg.get("enable_contrastive", False)),
            single_view=bool(train_cfg.get("single_view", False)),
        )

    return SwinUNetDualViewSSL(
        in_ch=int(model_cfg.get("in_ch", 1)),
        image_size=int(image_size),
        patch_size=int(model_cfg.get("patch_size", 16)),
        embed_dim=int(model_cfg.get("embed_dim", 96)),
        enc_depths=tuple(model_cfg.get("enc_depths", [2, 2, 6, 2])),
        dec_depths=tuple(model_cfg.get("dec_depths", [6, 2, 2])),
        num_heads=tuple(model_cfg.get("num_heads", [3, 6, 12, 24])),
        window_size=int(model_cfg.get("window_size", 7)),
        proj_dim=int(model_cfg.get("proj_dim", 128)),
        plane_inject_method=str(model_cfg.get("plane_inject_method", "film")),
        enable_saca=bool(model_cfg.get("enable_saca", False)),
        saca_position=str(model_cfg.get("saca_position", "after_stage1")),
        saca_positions=model_cfg.get("saca_positions", []),
        saca_gate_init=float(model_cfg.get("saca_gate_init", 0.0)),
        saca_warmup_epochs=int(model_cfg.get("saca_warmup_epochs", 5)),
        enable_reconstruct=bool(train_cfg.get("enable_reconstruct", True)),
        enable_contrastive=bool(train_cfg.get("enable_contrastive", False)),
        contrastive_loss_type=str(contrast_cfg.get("contrastive_loss_type", "infonce")),
        contrastive_position=str(contrast_cfg.get("contrastive_position", "bottleneck")),
        single_view=bool(train_cfg.get("single_view", False)),
    )


def _replace_recon_head_out_channels(model: torch.nn.Module, num_classes: int) -> None:
    import torch.nn as nn

    for attr in ("recon_head_v1", "recon_head_v2"):
        head = getattr(model, attr, None)
        if head is None:
            continue
        if not isinstance(head, nn.Sequential) or len(head) < 1 or not isinstance(head[-1], nn.Conv2d):
            continue
        last = head[-1]
        if int(last.out_channels) == int(num_classes):
            continue
        head[-1] = nn.Conv2d(last.in_channels, int(num_classes), kernel_size=1)


def _prepare_batch(
    batch: list[Sample],
    *,
    image_size: int,
    resize_mode: str,
    label_key: str,
    encoding_info,
    strict_label_ids: bool,
    allow_unknown_label_ids: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    planes: list[torch.Tensor] = []
    for s in batch:
        img_pil = load_image_pil(s.image_path)
        lbl_np = load_label_array(s.label_path, key=(label_key or None))
        x, y_ids = apply_pair_transforms(
            img_pil=img_pil,
            mask_array=lbl_np,
            image_size=int(image_size),
            do_hflip=False,
            resize_mode=resize_mode,
        )
        y_enc = encode_label_array(
            y_ids.squeeze(0).cpu().numpy(),
            encoding_info,
            strict_label_ids=bool(strict_label_ids),
            allow_unknown_label_ids=bool(allow_unknown_label_ids),
            unknown_fallback_id=0,
        )
        xs.append(x.to(dtype=torch.float32))
        ys.append(torch.from_numpy(y_enc).to(dtype=torch.long))
        plane = infer_plane_from_path(s.image_path, default_plane="axial")
        planes.append(plane_to_one_hot(plane).to(dtype=torch.float32))
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), torch.stack(planes, dim=0)


def _forward_logits(model: torch.nn.Module, x: torch.Tensor, plane: torch.Tensor) -> torch.Tensor:
    pixel_mask = torch.zeros((x.size(0), 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
    recon1, recon2, *_ = model(x, pixel_mask=pixel_mask, plane_one_hot=plane)
    recon2_aligned = flip_lr(recon2)
    return 0.5 * (recon1 + recon2_aligned)


def _per_image_macro_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    include_bg: bool,
) -> tuple[float, dict[int, float]]:
    per_class: dict[int, float] = {}
    vals: list[float] = []
    for cid in range(int(num_classes)):
        if (not include_bg) and cid == 0:
            continue
        tgt_c = target == cid
        pred_c = pred == cid
        denom = int(tgt_c.sum().item()) + int(pred_c.sum().item())
        if denom <= 0:
            continue
        inter = int((tgt_c & pred_c).sum().item())
        d = float((2.0 * inter + 1e-6) / (denom + 1e-6))
        per_class[cid] = d
        vals.append(d)
    if not vals:
        return float("nan"), per_class
    return float(np.mean(np.asarray(vals, dtype=np.float64))), per_class


def _build_class_name_map(id_to_name: dict[int, str], num_classes: int) -> dict[int, str]:
    out: dict[int, str] = {}
    for cid in range(int(num_classes)):
        out[cid] = str(id_to_name.get(cid, f"class_{cid}"))
    return out


def _draw_overlay(
    *,
    image: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    out_path: Path,
    class_name_map: dict[int, str],
    dice_value: float,
    include_bg: bool,
    dpi: int,
    linewidth: float,
    qmin: float,
    qmax: float,
) -> None:
    qmin_f = max(0.0, min(100.0, float(qmin)))
    qmax_f = max(0.0, min(100.0, float(qmax)))
    if qmax_f <= qmin_f:
        qmin_f, qmax_f = 1.0, 99.0
    vmin = float(np.percentile(image, qmin_f))
    vmax = float(np.percentile(image, qmax_f))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.0))
    ax_pred, ax_gt = axes
    ax_pred.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax_gt.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    cmap = plt.get_cmap("tab20")

    handles_pred = []
    handles_gt = []
    for cid in sorted(np.unique(pred).tolist()):
        if int(cid) < 0:
            continue
        if (not include_bg) and int(cid) == 0:
            continue
        mask = (pred == int(cid)).astype(np.float32)
        if int(mask.sum()) <= 0:
            continue
        color = cmap(int(cid) % cmap.N)
        ax_pred.contour(mask, levels=[0.5], colors=[color], linewidths=float(linewidth), alpha=1.0)
        handles_pred.append(plt.Line2D([0], [0], color=color, lw=2, label=f"{int(cid)}:{class_name_map.get(int(cid), f'class_{int(cid)}')}"))

    for cid in sorted(np.unique(target).tolist()):
        if int(cid) < 0:
            continue
        if (not include_bg) and int(cid) == 0:
            continue
        mask = (target == int(cid)).astype(np.float32)
        if int(mask.sum()) <= 0:
            continue
        color = cmap(int(cid) % cmap.N)
        ax_gt.contour(mask, levels=[0.5], colors=[color], linewidths=float(linewidth), alpha=1.0)
        handles_gt.append(plt.Line2D([0], [0], color=color, lw=2, label=f"{int(cid)}:{class_name_map.get(int(cid), f'class_{int(cid)}')}"))

    dice_txt = "nan" if (not math.isfinite(float(dice_value))) else f"{float(dice_value):.4f}"
    ax_pred.set_title(f"Pred boundary | dice={dice_txt}")
    ax_gt.set_title("Ground truth boundary")
    ax_pred.axis("off")
    ax_gt.axis("off")
    if handles_pred:
        ax_pred.legend(handles=handles_pred, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, frameon=True)
    if handles_gt:
        ax_gt.legend(handles=handles_gt, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _parse_id_csv(raw: str) -> set[int]:
    out: set[int] = set()
    for tok in str(raw).split(","):
        s = tok.strip()
        if not s:
            continue
        out.add(int(s))
    return out


def _should_exclude_from_ranking(
    target_ids: set[int],
    *,
    exclude_ids: set[int],
    ignore_ids: set[int],
    min_other_classes: int,
) -> bool:
    # Ignore background/ignore-index when deciding whether a sample is non-brain-only.
    effective = set(int(x) for x in target_ids if int(x) not in ignore_ids)
    if not effective:
        return True
    other = effective - exclude_ids
    return len(other) < max(0, int(min_other_classes))


def main() -> None:
    args = _parse_args()
    _mode4_guard(int(args.mode))

    image_root = Path(args.input_dir).expanduser().resolve()
    if not image_root.exists():
        raise FileNotFoundError(f"input-dir not found: {image_root}")
    label_root = image_root if args.label_dir is None else Path(args.label_dir).expanduser().resolve()
    if not label_root.exists():
        raise FileNotFoundError(f"label-dir not found: {label_root}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    label_suffixes = _resolve_label_suffixes(str(args.label_suffix), str(args.label_suffixes))
    print(f"[scan] source_match_label_suffix={str(args.label_suffix)}")
    if str(args.label_suffixes).strip():
        print(
            "[scan] note: --label-suffixes is ignored in source-match mode; "
            "pairing follows TissueSegmentationDataset using --label-suffix only."
        )
    print(
        "[labels] "
        f"mode={int(args.mode)} strict_label_ids={bool(args.strict_label_ids)} "
        f"allow_unknown_label_ids={bool(args.allow_unknown_label_ids)}"
    )

    seg_labels = parse_seg_labels_txt(args.seg_labels)
    unknown_ids, non_brain_ids = identify_special_ids(seg_labels)
    print(f"[labels] parsed_seg_labels={len(seg_labels)} from {Path(args.seg_labels).expanduser().resolve()}")
    if seg_labels:
        seg_id_min = min(int(k) for k in seg_labels.keys())
        seg_id_max = max(int(k) for k in seg_labels.keys())
        print(f"[labels] seg_id_range=[{seg_id_min},{seg_id_max}]")
    encoding_info = build_label_encoding_info(
        mode=int(args.mode),
        id_to_name=seg_labels,
        unknown_ids=unknown_ids,
        non_brain_ids=non_brain_ids,
        num_classes_override=int(args.num_classes),
        require_special_ids=False,
    )
    num_classes = int(encoding_info.num_classes)
    print(f"[labels] encoded_num_classes={num_classes}")
    class_name_map = _build_class_name_map(encoding_info.encoded_id_to_name, num_classes)

    samples = _collect_samples(
        dataset=TissueSegmentationDataset(
            image_root=image_root,
            label_root=label_root,
            scan_tokens=[
                str(p.relative_to(image_root)).replace("\\", "/")
                for p in sorted(image_root.rglob("*.png"))
                if p.is_file()
            ],
            encoding_info=encoding_info,
            image_ext=".png",
            label_suffix=str(args.label_suffix),
            label_key=(str(args.label_key) or None),
            image_size=int(args.image_size),
            target_size=int(args.image_size),
            resize_mode=str(args.resize_mode),
            plane="auto",
            strict_pairs=False,
            strict_label_ids=bool(args.strict_label_ids),
            allow_unknown_label_ids=bool(args.allow_unknown_label_ids),
            debug_shapes=False,
            image_index=build_image_index(image_root=image_root, image_ext=".png"),
        ),
        group_by_modality=bool(args.group_by_modality),
    )
    if not samples:
        raise RuntimeError(
            "No valid samples found. Ensure corresponding labels exist. "
            f"Tried suffixes: {label_suffixes}"
        )
    group_counts: dict[str, int] = {}
    for s in samples:
        group_counts[s.group] = group_counts.get(s.group, 0) + 1
    groups_present = sorted(group_counts.keys())
    print(f"[scan] groups_with_label={groups_present}")
    for g in groups_present:
        print(f"[scan/group] {g}: n={group_counts[g]}")

    device = _to_device(args.device)
    ckpt = _load_ckpt(Path(args.ckpt_path).expanduser().resolve(), device)
    ckpt_cfg = ckpt.get("cfg", {})
    if not isinstance(ckpt_cfg, dict):
        ckpt_cfg = {}

    image_size_ckpt = int(_get_cfg_value(ckpt_cfg, "data", "image_size", int(args.image_size)))
    model = _build_model_from_ckpt_cfg(
        ckpt_cfg,
        num_classes=num_classes,
        image_size=image_size_ckpt,
    )
    _replace_recon_head_out_channels(model, num_classes=num_classes)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)
    model.eval()

    records: list[dict[str, Any]] = []
    bs = max(1, int(args.batch_size))
    with torch.no_grad():
        for i in range(0, len(samples), bs):
            batch = samples[i : i + bs]
            x, y, plane = _prepare_batch(
                batch,
                image_size=int(args.image_size),
                resize_mode=str(args.resize_mode),
                label_key=str(args.label_key),
                encoding_info=encoding_info,
                strict_label_ids=bool(args.strict_label_ids),
                allow_unknown_label_ids=bool(args.allow_unknown_label_ids),
            )
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            plane = plane.to(device, non_blocking=True)
            logits = _forward_logits(model, x, plane)
            pred = torch.argmax(logits, dim=1)

            for j, s in enumerate(batch):
                dice_value, per_class_map = _per_image_macro_dice(
                    pred=pred[j],
                    target=y[j],
                    num_classes=num_classes,
                    include_bg=bool(args.include_bg),
                )
                records.append(
                    {
                        "image_path": str(s.image_path),
                        "label_path": str(s.label_path),
                        "group": s.group,
                        "dice": float(dice_value),
                        "pred": pred[j].detach().cpu().numpy().astype(np.int32),
                        "target": y[j].detach().cpu().numpy().astype(np.int32),
                        "image": x[j, 0].detach().cpu().numpy().astype(np.float32),
                        "per_class": per_class_map,
                    }
                )

    metrics_csv = out_dir / "dice_per_image.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label_path", "group", "dice"])
        w.writeheader()
        for r in records:
            w.writerow(
                {
                    "image_path": r["image_path"],
                    "label_path": r["label_path"],
                    "group": r["group"],
                    "dice": r["dice"],
                }
            )

    target_hist = np.zeros((num_classes,), dtype=np.int64)
    pred_hist = np.zeros((num_classes,), dtype=np.int64)
    for r in records:
        tgt = np.asarray(r["target"], dtype=np.int64).reshape(-1)
        prd = np.asarray(r["pred"], dtype=np.int64).reshape(-1)
        if tgt.size == 0:
            tgt_valid = np.zeros((0,), dtype=bool)
        else:
            tgt_valid = (tgt >= 0) & (tgt < num_classes)
        if prd.size == 0:
            prd_valid = np.zeros((0,), dtype=bool)
        else:
            prd_valid = (prd >= 0) & (prd < num_classes)
        if np.any(tgt_valid):
            target_hist += np.bincount(tgt[tgt_valid], minlength=num_classes).astype(np.int64)
        if np.any(prd_valid):
            pred_hist += np.bincount(prd[prd_valid], minlength=num_classes).astype(np.int64)

    target_hist_csv = out_dir / "target_class_hist.csv"
    with target_hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class_id", "class_name", "pixel_count"])
        w.writeheader()
        for cid in range(num_classes):
            w.writerow(
                {
                    "class_id": cid,
                    "class_name": class_name_map.get(cid, f"class_{cid}"),
                    "pixel_count": int(target_hist[cid]),
                }
            )
    pred_hist_csv = out_dir / "pred_class_hist.csv"
    with pred_hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class_id", "class_name", "pixel_count"])
        w.writeheader()
        for cid in range(num_classes):
            w.writerow(
                {
                    "class_id": cid,
                    "class_name": class_name_map.get(cid, f"class_{cid}"),
                    "pixel_count": int(pred_hist[cid]),
                }
            )

    nonzero = [(cid, int(target_hist[cid])) for cid in range(num_classes) if int(target_hist[cid]) > 0]
    nonzero_sorted = sorted(nonzero, key=lambda x: x[1], reverse=True)
    print(f"[target_hist] nonzero_classes={len(nonzero_sorted)}/{num_classes}")
    for cid, cnt in nonzero_sorted[:20]:
        print(f"[target_hist] c{cid}:{class_name_map.get(cid, f'class_{cid}')} pixels={cnt}")
    nonzero_pred = [(cid, int(pred_hist[cid])) for cid in range(num_classes) if int(pred_hist[cid]) > 0]
    nonzero_pred_sorted = sorted(nonzero_pred, key=lambda x: x[1], reverse=True)
    print(f"[pred_hist] nonzero_classes={len(nonzero_pred_sorted)}/{num_classes}")
    for cid, cnt in nonzero_pred_sorted[:20]:
        print(f"[pred_hist] c{cid}:{class_name_map.get(cid, f'class_{cid}')} pixels={cnt}")

    compare_csv = out_dir / "target_vs_pred_class_hist.csv"
    with compare_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "class_name",
                "target_pixel_count",
                "pred_pixel_count",
                "pred_over_target_ratio",
            ],
        )
        w.writeheader()
        for cid in range(num_classes):
            tcnt = int(target_hist[cid])
            pcnt = int(pred_hist[cid])
            if tcnt > 0:
                ratio = float(pcnt) / float(tcnt)
            else:
                ratio = float("nan")
            w.writerow(
                {
                    "class_id": cid,
                    "class_name": class_name_map.get(cid, f"class_{cid}"),
                    "target_pixel_count": tcnt,
                    "pred_pixel_count": pcnt,
                    "pred_over_target_ratio": ratio,
                }
            )

    summary_rows: list[dict[str, Any]] = []
    rank_exclude_ids = _parse_id_csv(str(args.exclude_rank_label_ids))
    rank_ignore_ids = {0, -100}
    print(f"[rank] exclude_if_target_only_in={sorted(rank_exclude_ids)}")
    print(f"[rank] ignore_ids_for_exclusion_check={sorted(rank_ignore_ids)}")
    print(f"[rank] min_other_classes={max(0, int(args.rank_min_other_classes))}")
    groups_for_summary = sorted(set(str(r["group"]) for r in records))
    for g in groups_for_summary:
        g_rows = [r for r in records if r["group"] == g]
        g_vals = [float(r["dice"]) for r in g_rows if math.isfinite(float(r["dice"]))]
        if g_vals:
            avg = float(np.mean(np.asarray(g_vals, dtype=np.float64)))
            std = float(np.std(np.asarray(g_vals, dtype=np.float64), ddof=0))
        else:
            avg = float("nan")
            std = float("nan")
        summary_rows.append(
            {
                "group": g,
                "num_samples": len(g_rows),
                "avg_dice": avg,
                "std_dice": std,
            }
        )

        rank_candidates: list[dict[str, Any]] = []
        excluded_for_rank = 0
        excluded_examples: list[str] = []
        for r in g_rows:
            tgt_ids = set(int(v) for v in np.unique(r["target"]).tolist())
            if _should_exclude_from_ranking(
                tgt_ids,
                exclude_ids=rank_exclude_ids,
                ignore_ids=rank_ignore_ids,
                min_other_classes=int(args.rank_min_other_classes),
            ):
                excluded_for_rank += 1
                if len(excluded_examples) < 3:
                    excluded_examples.append(
                        f"{Path(str(r['image_path'])).name}: target_ids={sorted(tgt_ids)}"
                    )
                continue
            rank_candidates.append(r)
        if excluded_for_rank > 0:
            print(f"[rank/{g}] excluded={excluded_for_rank} kept={len(rank_candidates)}")
            for ex in excluded_examples:
                print(f"[rank/{g}] excluded_example {ex}")

        ranked = sorted(
            rank_candidates,
            key=lambda x: float(x["dice"]) if math.isfinite(float(x["dice"])) else float("-inf"),
            reverse=True,
        )
        topk = ranked[: max(1, int(args.top_k))]
        for rank_idx, r in enumerate(topk, start=1):
            stem = Path(str(r["image_path"])).stem
            out_img = out_dir / "overlays" / g / f"rank_{rank_idx:02d}_dice_{float(r['dice']):.4f}_{stem}.png"
            _draw_overlay(
                image=r["image"],
                pred=r["pred"],
                target=r["target"],
                out_path=out_img,
                class_name_map=class_name_map,
                dice_value=float(r["dice"]),
                include_bg=bool(args.include_bg),
                dpi=int(args.overlay_dpi),
                linewidth=float(args.overlay_linewidth),
                qmin=float(args.overlay_contrast_qmin),
                qmax=float(args.overlay_contrast_qmax),
            )

    summary_csv = out_dir / "dice_group_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "num_samples", "avg_dice", "std_dice"])
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    print(f"[ok] mode={int(args.mode)} ckpt_load_mode={args.ckpt_load_mode}")
    print(f"[ok] samples={len(records)} num_classes={num_classes}")
    for row in summary_rows:
        avg_txt = "nan" if not math.isfinite(float(row["avg_dice"])) else f"{float(row['avg_dice']):.4f}"
        std_txt = "nan" if not math.isfinite(float(row["std_dice"])) else f"{float(row['std_dice']):.4f}"
        print(
            f"[group] {row['group']:<5} n={int(row['num_samples']):4d} "
            f"avg_dice={avg_txt} std_dice={std_txt}"
        )
    print(f"[out] per-image: {metrics_csv}")
    print(f"[out] target-hist: {target_hist_csv}")
    print(f"[out] pred-hist: {pred_hist_csv}")
    print(f"[out] target-vs-pred: {compare_csv}")
    print(f"[out] summary: {summary_csv}")
    print(f"[out] overlays: {out_dir / 'overlays'}")


if __name__ == "__main__":
    main()
