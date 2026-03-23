from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..skull_stripping.io import load_mask_array

DATASET_VERSION = "skull_stripping_preprocessed_v1"
META_FILENAME = "preprocess_meta.json"

@dataclass(frozen=True)
class SamplePair:
    image_path: Path
    mask_path: Path
    rel_image_path: Path

@dataclass(frozen=True)
class WorkerConfig:
    output_dir: Path
    output_ext: str
    output_mask_suffix: str
    overwrite: bool
    preserve_structure: bool
    image_height: int
    image_width: int
    resize_mode: str
    mask_key: str | None

def _normalize_ext(ext: str | None) -> str:
    if not ext:
        return ""
    out = ext.strip().lower()
    if not out:
        return ""
    if not out.startswith("."):
        out = f".{out}"
    return out

def _parse_image_size(values: Sequence[int]) -> tuple[int, int]:
    if len(values) == 1:
        side = int(values[0])
        if side <= 0:
            raise ValueError("--image-size must be positive")
        return side, side
    if len(values) == 2:
        h, w = int(values[0]), int(values[1])
        if h <= 0 or w <= 0:
            raise ValueError("--image-size H W must be positive")
        return h, w
    raise ValueError("--image-size accepts either one value (S) or two values (H W)")

def _resize_pair_tensors(
    image: torch.Tensor,
    mask: torch.Tensor,
    out_h: int,
    out_w: int,
    resize_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:

    if resize_mode == "direct":
        img_r = F.interpolate(image, size=(out_h, out_w), mode="bilinear", align_corners=False)
        mask_r = F.interpolate(mask, size=(out_h, out_w), mode="nearest")
        return img_r, mask_r

    if resize_mode != "letterbox":
        raise ValueError(f"Unsupported resize_mode={resize_mode}. Expected 'direct' or 'letterbox'.")

    src_h, src_w = int(image.shape[-2]), int(image.shape[-1])
    scale = min(out_w / float(src_w), out_h / float(src_h))
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    img_r = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
    mask_r = F.interpolate(mask, size=(new_h, new_w), mode="nearest")

    image_out = torch.zeros((1, 1, out_h, out_w), dtype=image.dtype)
    mask_out = torch.zeros((1, 1, out_h, out_w), dtype=mask.dtype)
    left = (out_w - new_w) // 2
    top = (out_h - new_h) // 2
    image_out[:, :, top : top + new_h, left : left + new_w] = img_r
    mask_out[:, :, top : top + new_h, left : left + new_w] = mask_r
    return image_out, mask_out

def iter_samples(input_dir: Path, image_ext: str, mask_suffix: str, strict_pairs: bool) -> Iterable[SamplePair]:
    root = input_dir.resolve()
    img_ext = _normalize_ext(image_ext)
    all_images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == img_ext)
    missing: list[Path] = []
    for image_path in all_images:
        mask_path = image_path.with_name(f"{image_path.stem}{mask_suffix}")
        if mask_path.exists():
            yield SamplePair(
                image_path=image_path,
                mask_path=mask_path,
                rel_image_path=image_path.relative_to(root),
            )
            continue
        missing.append(image_path)

    if strict_pairs and missing:
        sample = ", ".join(p.name for p in missing[:5])
        raise FileNotFoundError(
            f"Missing masks for {len(missing)} images (e.g., {sample}). "
            f"Expected '{mask_suffix}' suffix next to each {img_ext} image."
        )

def _make_output_rel_path(rel_image_path: Path, preserve_structure: bool) -> Path:
    if preserve_structure:
        return rel_image_path
    if len(rel_image_path.parts) <= 1:
        return Path(rel_image_path.name)
    stem = "__".join(rel_image_path.with_suffix("").parts)
    return Path(f"{stem}{rel_image_path.suffix}")

def _load_image_gray(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        if img.mode != "L":
            img = img.convert("L")
        return np.asarray(img, dtype=np.float32) / 255.0

def _resolve_image_ext(output_ext: str, source_path: Path) -> str:
    if output_ext:
        return output_ext
    return source_path.suffix.lower()

def _save_mask(mask_path: Path, mask_array: np.ndarray) -> None:
    if mask_path.suffix == ".npy":
        np.save(mask_path, mask_array, allow_pickle=False)
        return
    if mask_path.suffix == ".npz":
        np.savez(mask_path, mask=mask_array)
        return
    raise ValueError(f"Unsupported mask output extension: {mask_path.suffix}. Use .npy or .npz")

def process_one(sample: SamplePair, cfg: WorkerConfig) -> tuple[int, int]:
    out_rel = _make_output_rel_path(sample.rel_image_path, preserve_structure=cfg.preserve_structure)
    out_ext = _resolve_image_ext(cfg.output_ext, source_path=sample.image_path)
    out_image_path = (cfg.output_dir / out_rel).with_suffix(out_ext)
    out_mask_path = out_image_path.with_name(f"{out_image_path.stem}{cfg.output_mask_suffix}")
    out_image_path.parent.mkdir(parents=True, exist_ok=True)

    if out_image_path.exists() and out_mask_path.exists() and not cfg.overwrite:
        return (0, 1)

    img_np = _load_image_gray(sample.image_path)
    mask_np = load_mask_array(sample.mask_path, key=cfg.mask_key)

    img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    mask_t = torch.from_numpy(mask_np).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    img_t, mask_t = _resize_pair_tensors(
        img_t,
        mask_t,
        out_h=cfg.image_height,
        out_w=cfg.image_width,
        resize_mode=cfg.resize_mode,
    )

    img_out = img_t.squeeze(0).squeeze(0).clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8).cpu().numpy()
    mask_i64 = mask_t.squeeze(0).squeeze(0).round().to(dtype=torch.int64)
    mask_min = int(mask_i64.min().item())
    mask_max = int(mask_i64.max().item())
    if mask_min < 0 or mask_max > 255:
        raise ValueError(
            f"Mask ids out of uint8 range for {sample.mask_path}: min={mask_min}, max={mask_max}. "
            "Regenerate with a wider mask dtype if needed."
        )
    mask_out = mask_i64.to(dtype=torch.uint8).cpu().numpy()

    Image.fromarray(img_out, mode="L").save(out_image_path)
    _save_mask(out_mask_path, mask_out)
    return (1, 0)

def write_meta(
    output_dir: Path,
    *,
    image_height: int,
    image_width: int,
    image_ext: str,
    mask_suffix: str,
    resize_mode: str,
    input_image_ext: str,
    input_mask_suffix: str,
    total_pairs: int,
) -> Path:
    meta = {
        "dataset_version": DATASET_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_size": {"height": image_height, "width": image_width},
        "resize_mode": resize_mode,
        "image_ext": image_ext,
        "mask_suffix": mask_suffix,
        "input_image_ext": input_image_ext,
        "input_mask_suffix": input_mask_suffix,
        "input_dtype_range": "uint8_on_disk -> float32 [0,1] in loader",
        "target_dtype_range": "uint8_ids_on_disk -> float32 target/255.0 in loader (or (target>0).float() with --binarize-target)",
        "normalization": {
            "input": "x = uint8 / 255.0",
            "target": "y = ids / 255.0 unless binarize_target is enabled in training",
        },
        "total_pairs": int(total_pairs),
    }
    out_path = output_dir / META_FILENAME
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_path

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m swin_unet.src.ver4.tools.preprocess_skull_stripping_dataset",
        description="Offline preprocessing for ver4 skull stripping image-mask pairs.",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Input folder containing image/mask pairs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder for preprocessed dataset.")
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        required=True,
        metavar="SIZE",
        help="Output size: pass one value (S) or two values (H W).",
    )
    parser.add_argument("--ext", type=str, default="", help="Output image extension (e.g. .png). Empty keeps source extension.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of worker processes. 0/1 uses single process.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing preprocessed files.")
    parser.add_argument("--preserve-structure", action="store_true", help="Preserve relative path structure under output-dir.")

    parser.add_argument("--input-image-ext", type=str, default=".png", help="Input image extension to scan (default: .png).")
    parser.add_argument("--mask-suffix", type=str, default="_mask.npz", help="Input mask suffix next to each image stem.")
    parser.add_argument("--output-mask-suffix", type=str, default="_mask.npy", help="Output mask suffix (.npy or .npz).")
    parser.add_argument("--mask-key", type=str, default="", help="Optional key when reading NPZ masks.")
    parser.add_argument("--resize-mode", type=str, default="letterbox", choices=["letterbox", "direct"], help="Resize strategy.")
    parser.add_argument("--strict-pairs", type=int, default=1, help="1: error on missing mask; 0: skip missing.")
    return parser.parse_args()

def main() -> None:
    args = _parse_args()

    image_h, image_w = _parse_image_size(args.image_size)
    output_ext = _normalize_ext(args.ext)
    input_image_ext = _normalize_ext(args.input_image_ext)
    output_mask_suffix = args.output_mask_suffix.strip()
    if output_mask_suffix in {".npy", ".npz"}:
        output_mask_suffix = f"_mask{output_mask_suffix}"
    if not output_mask_suffix.startswith("_"):
        output_mask_suffix = f"_{output_mask_suffix}"
    if not output_mask_suffix.endswith((".npy", ".npz")):
        raise ValueError("--output-mask-suffix must end with .npy or .npz")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"input-dir not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = list(iter_samples(input_dir, input_image_ext, args.mask_suffix, strict_pairs=bool(args.strict_pairs)))
    if not samples:
        raise RuntimeError(f"No valid image/mask pairs found under {input_dir}")

    worker_cfg = WorkerConfig(
        output_dir=output_dir,
        output_ext=output_ext,
        output_mask_suffix=output_mask_suffix,
        overwrite=bool(args.overwrite),
        preserve_structure=bool(args.preserve_structure),
        image_height=image_h,
        image_width=image_w,
        resize_mode=args.resize_mode,
        mask_key=(args.mask_key or None),
    )

    written = 0
    skipped = 0
    workers = max(0, int(args.num_workers))
    if workers <= 1:
        for sample in samples:
            w, s = process_one(sample, worker_cfg)
            written += w
            skipped += s
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_one, sample, worker_cfg) for sample in samples]
            for future in as_completed(futures):
                w, s = future.result()
                written += w
                skipped += s

    resolved_image_ext = output_ext if output_ext else "mixed_by_source"
    meta_path = write_meta(
        output_dir=output_dir,
        image_height=image_h,
        image_width=image_w,
        image_ext=resolved_image_ext,
        mask_suffix=output_mask_suffix,
        resize_mode=args.resize_mode,
        input_image_ext=input_image_ext,
        input_mask_suffix=args.mask_suffix,
        total_pairs=len(samples),
    )

    print(f"[preprocess] input={input_dir}")
    print(f"[preprocess] output={output_dir}")
    print(f"[preprocess] pairs={len(samples)} written={written} skipped={skipped}")
    print(f"[preprocess] image_size=({image_h}, {image_w}) resize_mode={args.resize_mode}")
    print(f"[preprocess] metadata={meta_path}")

if __name__ == "__main__":
    main()
