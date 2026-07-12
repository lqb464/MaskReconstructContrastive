#!/usr/bin/env python3
"""
prepare_brats2021.py

Convert BraTS 2021 volumetric NIfTI data to 2D PNG slices + NPZ label files
compatible with tumor_segmentation / TissueSegmentationDataset.

Supports:
  - Extracted patient folders under --brats-root
  - .tar / .tar.gz archives passed directly to --brats-root (common on Kaggle)

Expected structure inside the archive/root:
    BraTS2021_00000/
        BraTS2021_00000_flair.nii.gz
        BraTS2021_00000_t1.nii.gz
        BraTS2021_00000_t1ce.nii.gz
        BraTS2021_00000_t2.nii.gz
        BraTS2021_00000_seg.nii.gz
    BraTS2021_00001/
        ...

Output structure:
    <out-root>/
        images/
            BraTS2021_00000_z0080.png
        labels/
            BraTS2021_00000_z0080_label.npz
        train_list.txt
        eval_list.txt

Usage (Kaggle, tar input):
    python autoencoder/scripts/prepare_brats2021.py \\
        --brats-root /kaggle/input/.../BraTS2021_Training_Data.tar \\
        --out-root   /data/brats2021_2d \\
        --modality   flair \\
        --val-ratio  0.15
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


def _is_tar_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in _TAR_SUFFIXES)


def _extract_tar_archive(archive_path: Path, extract_dir: Path) -> Path:
    """
    Extract a BraTS tar archive and return the directory containing patient folders.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[extract] extracting {archive_path} -> {extract_dir}")

    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(path=extract_dir)

    # Find directory that contains patient subfolders (BraTS2021_*)
    candidates = [extract_dir]
    candidates.extend(sorted(p for p in extract_dir.iterdir() if p.is_dir()))

    for candidate in candidates:
        patient_dirs = [
            p for p in candidate.iterdir()
            if p.is_dir() and p.name.startswith("BraTS")
        ]
        if patient_dirs:
            log.info(
                f"[extract] found {len(patient_dirs)} patient folders under {candidate}"
            )
            return candidate

    raise RuntimeError(
        f"Extracted archive at {extract_dir} but no BraTS patient folders were found. "
        "Check that --brats-root points to BraTS2021_Training_Data.tar."
    )


def resolve_brats_root(brats_root: str | Path, *, work_dir: Optional[Path] = None) -> tuple[Path, Optional[Path]]:
    """
    Resolve --brats-root to a directory with patient folders.

    Returns:
        (data_root, temp_dir_to_cleanup_or_None)
    """
    root = Path(brats_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"--brats-root not found: {root}")

    if root.is_file() and _is_tar_archive(root):
        temp_parent = work_dir if work_dir is not None else Path(tempfile.mkdtemp(prefix="brats2021_extract_"))
        temp_parent.mkdir(parents=True, exist_ok=True)
        extract_target = temp_parent / "extracted"
        data_root = _extract_tar_archive(root, extract_target)
        return data_root, temp_parent

    if root.is_dir():
        patient_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("BraTS")]
        if patient_dirs:
            return root, None

        # Maybe one extra nesting level: BraTS2021_Training_Data/BraTS2021_00000/...
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            nested = [p for p in child.iterdir() if p.is_dir() and p.name.startswith("BraTS")]
            if nested:
                log.info(f"[brats-root] using nested data directory: {child}")
                return child, None

    raise RuntimeError(
        f"No BraTS patient folders found under --brats-root={root}. "
        "Expected directories like BraTS2021_00000/ with *_flair.nii.gz and *_seg.nii.gz."
    )


def _load_nifti(path: Path) -> np.ndarray:
    """Load NIfTI file, orient to RAI, and return float32 volume [H, W, D]."""
    try:
        import SimpleITK as sitk  # type: ignore
        img = sitk.ReadImage(str(path))
        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation("RAI")
        img = orienter.Execute(img)
        vol = sitk.GetArrayFromImage(img).astype(np.float32)
        return np.transpose(vol, (1, 2, 0))  # [Z, Y, X] -> [Y, X, Z]
    except ImportError:
        pass

    try:
        import nibabel as nib  # type: ignore
        img = nib.load(str(path))
        orig_ornt = nib.orientations.io_orientation(img.affine)
        targ_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'I'))
        transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
        img_oriented = img.as_reoriented(transform)
        return np.asarray(img_oriented.dataobj, dtype=np.float32)
    except (ImportError, Exception):
        pass

    try:
        import nibabel as nib  # type: ignore
        img = nib.load(str(path))
        return np.asarray(img.dataobj, dtype=np.float32)
    except ImportError:
        pass

    raise RuntimeError(
        f"Could not load {path}: install nibabel or SimpleITK "
        "(pip install nibabel  OR  pip install SimpleITK)."
    )


def _load_seg_nifti(path: Path) -> np.ndarray:
    """Load segmentation NIfTI, orient to RAI, and return int16 volume [H, W, D]."""
    try:
        import SimpleITK as sitk  # type: ignore
        img = sitk.ReadImage(str(path))
        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation("RAI")
        img = orienter.Execute(img)
        vol = sitk.GetArrayFromImage(img).astype(np.int16)
        return np.transpose(vol, (1, 2, 0))
    except ImportError:
        pass

    try:
        import nibabel as nib  # type: ignore
        img = nib.load(str(path))
        orig_ornt = nib.orientations.io_orientation(img.affine)
        targ_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'I'))
        transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
        img_oriented = img.as_reoriented(transform)
        return np.asarray(img_oriented.dataobj).astype(np.int16)
    except (ImportError, Exception):
        pass

    try:
        import nibabel as nib  # type: ignore
        img = nib.load(str(path))
        return np.asarray(img.dataobj).astype(np.int16)
    except ImportError:
        pass

    raise RuntimeError(
        "Could not load segmentation NIfTI: install nibabel or SimpleITK."
    )


def _normalize_slice_to_uint8(sl: np.ndarray) -> np.ndarray:
    sl = sl.astype(np.float32)
    mn, mx = float(sl.min()), float(sl.max())
    if mx - mn < 1e-8:
        return np.zeros_like(sl, dtype=np.uint8)
    sl = (sl - mn) / (mx - mn) * 255.0
    return np.clip(sl, 0, 255).astype(np.uint8)


def _save_png(arr: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image  # type: ignore
        Image.fromarray(arr, mode="L").save(str(path))
        return
    except ImportError:
        pass

    try:
        import cv2  # type: ignore
        cv2.imwrite(str(path), arr)
        return
    except ImportError:
        pass

    raise RuntimeError(
        "Cannot save PNG: install Pillow or opencv-python."
    )


def _is_empty_label(label_sl: np.ndarray) -> bool:
    return bool((label_sl == 0).all())


def process_patient(
    patient_dir: Path,
    *,
    modality: str,
    out_images: Path,
    out_labels: Path,
    bg_keep_prob: float,
    rng: random.Random,
    z_range: Optional[Tuple[int, int]] = None,
    num_slices: int = 0,
) -> List[str]:
    patient_id = patient_dir.name

    img_candidates = [
        patient_dir / f"{patient_id}_{modality}.nii.gz",
        patient_dir / f"{patient_id}_{modality}.nii",
    ]
    seg_candidates = [
        patient_dir / f"{patient_id}_seg.nii.gz",
        patient_dir / f"{patient_id}_seg.nii",
    ]

    img_path = next((p for p in img_candidates if p.exists()), None)
    seg_path = next((p for p in seg_candidates if p.exists()), None)

    if img_path is None:
        log.warning(f"[skip] {patient_id}: modality file not found ({modality})")
        return []
    if seg_path is None:
        log.warning(f"[skip] {patient_id}: seg file not found")
        return []

    log.info(f"[process] {patient_id} modality={modality}")

    img_vol = _load_nifti(img_path)
    seg_vol = _load_seg_nifti(seg_path)

    if img_vol.shape != seg_vol.shape:
        log.warning(
            f"[skip] {patient_id}: shape mismatch img={img_vol.shape} seg={seg_vol.shape}"
        )
        return []

    _, _, depth = img_vol.shape
    z_start = z_range[0] if z_range else 0
    z_end = z_range[1] if z_range else depth
    z_end = min(z_end, depth)

    # Filter slices first to find non-empty/valid ones
    valid_zs: List[int] = []
    for z in range(z_start, z_end):
        img_sl = img_vol[:, :, z]
        seg_sl = seg_vol[:, :, z]

        # Skip if the image slice is empty of brain tissue (all black/zero)
        # Check if there are at least 500 non-zero voxels (approx 0.8% of 240x240 image)
        if (img_sl > 0).sum() < 500:
            continue

        # If slice has no tumor, keep with probability bg_keep_prob
        if _is_empty_label(seg_sl):
            if rng.random() > bg_keep_prob:
                continue

        valid_zs.append(z)

    # Select from valid_zs
    if num_slices > 0:
        if len(valid_zs) == 0:
            log.warning(f"[warn] {patient_id}: no non-empty slices found")
            z_indices = []
        elif len(valid_zs) <= num_slices:
            z_indices = valid_zs
        else:
            indices = np.linspace(0, len(valid_zs) - 1, num_slices, dtype=int)
            z_indices = [valid_zs[i] for i in indices]
    else:
        z_indices = valid_zs

    written: List[str] = []
    for z in z_indices:
        img_sl = img_vol[:, :, z]
        seg_sl = seg_vol[:, :, z]

        stem = f"{patient_id}_z{z:04d}"
        _save_png(_normalize_slice_to_uint8(img_sl), out_images / f"{stem}.png")
        np.savez_compressed(str(out_labels / f"{stem}_label.npz"), label=seg_sl.astype(np.int16))
        written.append(stem)

    return written


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BraTS 2021 NIfTI volumes to 2D PNG slices for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--brats-root",
        required=True,
        help=(
            "Path to extracted BraTS2021_Training_Data directory OR a .tar/.tar.gz archive "
            "(Kaggle dataset path is supported)."
        ),
    )
    parser.add_argument("--out-root", required=True, help="Output root directory.")
    parser.add_argument(
        "--modality",
        default="flair",
        choices=["flair", "t1", "t1ce", "t2"],
        help="MRI modality to extract as input image.",
    )
    parser.add_argument(
        "--bg-keep-prob",
        type=float,
        default=0.10,
        help="Fraction of all-background slices to keep (1.0 keeps all).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of patients for eval split (patient-level).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--z-range",
        type=str,
        default="",
        help="Optional z-slice range 'start:end' (e.g. '40:130').",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=0,
        help="Number of slices to extract per patient (0 to keep all based on bg-keep-prob).",
    )
    parser.add_argument(
        "--patients",
        type=str,
        default="",
        help="Comma-separated patient IDs to process (default: all).",
    )
    parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle before split.")
    parser.add_argument(
        "--extract-dir",
        type=str,
        default="",
        help="Directory to extract tar into (default: temp dir; use /kaggle/working/brats_extract on Kaggle).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    out_root = Path(args.out_root).expanduser().resolve()
    out_images = out_root / "images"
    out_labels = out_root / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    z_range: Optional[Tuple[int, int]] = None
    if args.z_range.strip():
        parts = args.z_range.strip().split(":")
        if len(parts) != 2:
            log.error(f"--z-range must be 'start:end', got: {args.z_range!r}")
            sys.exit(1)
        z_range = (int(parts[0]), int(parts[1]))
        log.info(f"[z-range] restricting to z in [{z_range[0]}, {z_range[1]})")

    work_dir = Path(args.extract_dir).expanduser().resolve() if args.extract_dir.strip() else None
    data_root, temp_dir = resolve_brats_root(args.brats_root, work_dir=work_dir)

    try:
        if args.patients.strip():
            patient_ids = [p.strip() for p in args.patients.split(",") if p.strip()]
            patient_dirs = []
            for pid in patient_ids:
                p = data_root / pid
                if p.is_dir():
                    patient_dirs.append(p)
                else:
                    log.warning(f"Patient dir not found: {p}")
        else:
            patient_dirs = sorted(
                p for p in data_root.iterdir()
                if p.is_dir() and p.name.startswith("BraTS")
            )

        if not patient_dirs:
            log.error(f"No patient directories found under {data_root}")
            sys.exit(1)

        log.info(f"[patients] found {len(patient_dirs)} patient directories")

        rng_split = random.Random(args.seed)
        rng_slice = random.Random(args.seed + 1)

        patient_dirs_ordered = list(patient_dirs)
        if not args.no_shuffle:
            rng_split.shuffle(patient_dirs_ordered)

        n_val = max(1, int(round(len(patient_dirs_ordered) * args.val_ratio)))
        eval_dirs = {p.name for p in patient_dirs_ordered[:n_val]}
        train_dirs = {p.name for p in patient_dirs_ordered[n_val:]}

        log.info(
            f"[split] train_patients={len(train_dirs)} eval_patients={len(eval_dirs)} "
            f"val_ratio={args.val_ratio:.2f}"
        )

        train_tokens: List[str] = []
        eval_tokens: List[str] = []
        total_written = 0

        for patient_dir in patient_dirs_ordered:
            stems = process_patient(
                patient_dir,
                modality=args.modality,
                out_images=out_images,
                out_labels=out_labels,
                bg_keep_prob=args.bg_keep_prob,
                rng=rng_slice,
                z_range=z_range,
                num_slices=args.num_slices,
            )
            total_written += len(stems)

            patient_token = patient_dir.name
            if patient_dir.name in eval_dirs:
                eval_tokens.append(patient_token)
            else:
                train_tokens.append(patient_token)

            log.info(f"  {patient_dir.name}: wrote {len(stems)} slices")

        train_list_path = out_root / "train_list.txt"
        eval_list_path = out_root / "eval_list.txt"
        train_list_path.write_text("\n".join(sorted(train_tokens)) + "\n", encoding="utf-8")
        eval_list_path.write_text("\n".join(sorted(eval_tokens)) + "\n", encoding="utf-8")

        log.info(f"\n[done] total slices written: {total_written}")
        log.info(f"[done] images      -> {out_images}")
        log.info(f"[done] labels      -> {out_labels}")
        log.info(f"[done] train_list  -> {train_list_path} ({len(train_tokens)} patients)")
        log.info(f"[done] eval_list   -> {eval_list_path} ({len(eval_tokens)} patients)")
        log.info("")
        log.info("Next: train with")
        log.info("  python -m autoencoder.src.tumor_segmentation.main \\")
        log.info(f"    --train-root {out_images} --train-label {out_labels} \\")
        log.info(f"    --eval-root {out_images} --eval-label {out_labels} \\")
        log.info(f"    --train-list {train_list_path} --eval-list {eval_list_path} \\")
        log.info("    --backbone vae --epochs 50 --out-dir autoencoder/outputs/brats2021/tumor_seg")

    finally:
        if temp_dir is not None and not args.extract_dir.strip():
            log.info(f"[extract] removing temporary extract dir {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
