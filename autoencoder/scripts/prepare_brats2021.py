#!/usr/bin/env python3
"""
prepare_brats2021.py

Convert BraTS 2021 volumetric NIfTI data to 2D PNG slices + NPZ label files
compatible with tumor_segmentation / TissueSegmentationDataset.

Slice selection reuses the ADNI energy-threshold logic (extract_brain_slices).
The same slice indices are applied to every requested modality and to seg.

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

Output structure (--modality all):
    <out-root>/
        images/
            BraTS2021_00000_t1_z0080.png
            BraTS2021_00000_t1ce_z0080.png
            BraTS2021_00000_t2_z0080.png
            BraTS2021_00000_flair_z0080.png
        labels/
            BraTS2021_00000_t1_z0080_label.npz
            ... (same mask stem-matched per modality)
        train_list.txt
        eval_list.txt

Usage (Kaggle, all 4 modalities + seg):
    python autoencoder/scripts/prepare_brats2021.py \\
        --brats-root /kaggle/input/.../BraTS2021_Training_Data.tar \\
        --extract-dir /kaggle/working/brats_extract \\
        --out-root   /kaggle/working/brats2021_2d \\
        --modality   all \\
        --num-slices 20 \\
        --val-ratio  0.2
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

log = logging.getLogger(__name__)

_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
ALL_MODALITIES = ("t1", "t1ce", "t2", "flair")


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


def _load_nifti_zyx(path: Path, *, dtype=np.float32) -> np.ndarray:
    """
    Load NIfTI, orient to RAI, return array shaped (Z, H, W) — same convention as ADNI/SimpleITK.
    """
    try:
        import SimpleITK as sitk  # type: ignore

        sitk.ProcessObject_SetGlobalWarningDisplay(False)
        img = sitk.ReadImage(str(path))
        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation("RAI")
        img = orienter.Execute(img)
        return sitk.GetArrayFromImage(img).astype(dtype)
    except ImportError:
        pass

    try:
        import nibabel as nib  # type: ignore

        img = nib.load(str(path))
        orig_ornt = nib.orientations.io_orientation(img.affine)
        targ_ornt = nib.orientations.axcodes2ornt(("R", "A", "I"))
        transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
        img_oriented = img.as_reoriented(transform)
        # nibabel data is typically (X, Y, Z); move Z to axis 0 to match SimpleITK.
        vol = np.asarray(img_oriented.dataobj, dtype=dtype)
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
        return np.transpose(vol, (2, 1, 0))
    except (ImportError, Exception):
        pass

    raise RuntimeError(
        f"Could not load {path}: install SimpleITK or nibabel "
        "(pip install SimpleITK  OR  pip install nibabel)."
    )


def extract_brain_slices(volume_np: np.ndarray, n_slices: int = 20):
    """
    ADNI slice picker (kept as-is in spirit).

    volume_np: numpy array of shape (Z, H, W)
    returns: list of 2D slices, slice indices, (start, end) brain region
    """
    Z = volume_np.shape[0]

    energy = volume_np.reshape(Z, -1).mean(axis=1)
    energy_smooth = gaussian_filter1d(energy, sigma=5)

    threshold = energy_smooth.min() + 0.4 * (energy_smooth.max() - energy_smooth.min())
    brain_mask = energy_smooth > threshold

    idx = np.where(brain_mask)[0]
    if len(idx) == 0:
        start, end = int(Z * 0.25), int(Z * 0.65)
    else:
        start, end = idx[0], (idx[-1] - idx[0]) // 2 + idx[0]

    slice_indices = np.linspace(start, end, n_slices, dtype=int)
    slices = [volume_np[i] for i in slice_indices]
    return slices, slice_indices, (start, end)


def _normalize_slices_to_uint8(slices: Sequence[np.ndarray]) -> List[np.ndarray]:
    """ADNI-style shared min/max normalize across the selected slices of one volume."""
    img_max = max(float(s.max()) for s in slices)
    img_min = min(float(s.min()) for s in slices)
    out: List[np.ndarray] = []
    for s in slices:
        img = s.astype(np.float32)
        img = img - img_min
        img = img / (img_max + 1e-5)
        out.append((img * 255.0).astype(np.uint8))
    return out


def _save_png(arr: np.ndarray, path: Path) -> None:
    try:
        import imageio  # type: ignore

        imageio.imwrite(str(path), arr)
        return
    except ImportError:
        pass

    try:
        from PIL import Image  # type: ignore

        Image.fromarray(arr, mode="L").save(str(path))
        return
    except ImportError:
        pass

    raise RuntimeError("Cannot save PNG: install imageio or Pillow.")


def _resolve_patient_file(patient_dir: Path, patient_id: str, suffix: str) -> Optional[Path]:
    candidates = [
        patient_dir / f"{patient_id}_{suffix}.nii.gz",
        patient_dir / f"{patient_id}_{suffix}.nii",
    ]
    return next((p for p in candidates if p.exists()), None)


def parse_modalities(modality_arg: str) -> List[str]:
    raw = modality_arg.strip().lower()
    if raw in {"all", "*"}:
        return list(ALL_MODALITIES)
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--modality is empty")
    unknown = [p for p in parts if p not in ALL_MODALITIES]
    if unknown:
        raise ValueError(
            f"Unknown modality(ies): {unknown}. "
            f"Choose from {list(ALL_MODALITIES)} or 'all'."
        )
    # preserve order, drop duplicates
    seen = set()
    ordered: List[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def process_patient(
    patient_dir: Path,
    *,
    modalities: Sequence[str],
    out_images: Path,
    out_labels: Path,
    num_slices: int,
    energy_modality: str,
) -> Tuple[str, int, Optional[str]]:
    """
    Extract synced slices for all modalities + seg.

    Returns: (patient_id, num_stems_written, error_or_None)
    """
    patient_id = patient_dir.name
    try:
        seg_path = _resolve_patient_file(patient_dir, patient_id, "seg")
        if seg_path is None:
            return patient_id, 0, "seg file not found"

        mod_paths: dict[str, Path] = {}
        for mod in modalities:
            p = _resolve_patient_file(patient_dir, patient_id, mod)
            if p is None:
                return patient_id, 0, f"modality file not found ({mod})"
            mod_paths[mod] = p

        if energy_modality not in mod_paths:
            return patient_id, 0, f"energy modality not in requested set: {energy_modality}"

        energy_vol = _load_nifti_zyx(mod_paths[energy_modality], dtype=np.float32)
        _, slice_indices, _ = extract_brain_slices(energy_vol, n_slices=num_slices)

        seg_vol = _load_nifti_zyx(seg_path, dtype=np.int16)
        if seg_vol.shape != energy_vol.shape:
            return (
                patient_id,
                0,
                f"shape mismatch energy={energy_vol.shape} seg={seg_vol.shape}",
            )

        mod_vols: dict[str, np.ndarray] = {energy_modality: energy_vol}
        for mod, path in mod_paths.items():
            if mod == energy_modality:
                continue
            vol = _load_nifti_zyx(path, dtype=np.float32)
            if vol.shape != energy_vol.shape:
                return (
                    patient_id,
                    0,
                    f"shape mismatch {mod}={vol.shape} energy={energy_vol.shape}",
                )
            mod_vols[mod] = vol

        # Shared min/max normalize per modality across selected slices (ADNI-style).
        mod_uint8: dict[str, List[np.ndarray]] = {}
        for mod in modalities:
            selected = [mod_vols[mod][int(z)] for z in slice_indices]
            mod_uint8[mod] = _normalize_slices_to_uint8(selected)

        written = 0
        for i, z in enumerate(slice_indices):
            z = int(z)
            seg_sl = seg_vol[z]
            for mod in modalities:
                stem = f"{patient_id}_{mod}_z{z:04d}"
                _save_png(mod_uint8[mod][i], out_images / f"{stem}.png")
                np.savez_compressed(
                    str(out_labels / f"{stem}_label.npz"),
                    label=seg_sl.astype(np.int16),
                )
                written += 1

        return patient_id, written, None
    except Exception as e:
        return patient_id, 0, str(e)


def _process_patient_job(args: tuple) -> Tuple[str, int, Optional[str]]:
    (
        patient_dir_str,
        modalities,
        out_images_str,
        out_labels_str,
        num_slices,
        energy_modality,
    ) = args
    return process_patient(
        Path(patient_dir_str),
        modalities=modalities,
        out_images=Path(out_images_str),
        out_labels=Path(out_labels_str),
        num_slices=num_slices,
        energy_modality=energy_modality,
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert BraTS 2021 NIfTI volumes to 2D PNG slices + synced seg labels. "
            "Supports one modality or all four via --modality all."
        ),
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
        default="all",
        help=(
            "MRI modality to extract: one of {t1,t1ce,t2,flair}, "
            "comma-separated list, or 'all' for all four."
        ),
    )
    parser.add_argument(
        "--energy-modality",
        default="",
        help=(
            "Modality used for ADNI extract_brain_slices energy curve. "
            "Default: first modality in --modality (for 'all' that is t1)."
        ),
    )
    parser.add_argument(
        "--bg-keep-prob",
        type=float,
        default=0.10,
        help="Unused with ADNI brain-slice picker; kept for CLI compatibility.",
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
        help="Unused with ADNI brain-slice picker; kept for CLI compatibility.",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=20,
        help="Number of slices to extract per patient (ADNI extract_brain_slices).",
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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Process pool workers (1 = sequential).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        modalities = parse_modalities(args.modality)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)

    energy_modality = args.energy_modality.strip().lower() or modalities[0]
    if energy_modality not in modalities:
        log.error(
            f"--energy-modality={energy_modality!r} must be one of selected modalities {modalities}"
        )
        sys.exit(1)

    if args.num_slices <= 0:
        log.error("--num-slices must be > 0 when using ADNI extract_brain_slices")
        sys.exit(1)

    if args.bg_keep_prob != 0.10 or args.z_range.strip():
        log.warning(
            "[compat] --bg-keep-prob / --z-range are ignored; "
            "slice indices come from ADNI extract_brain_slices."
        )

    out_root = Path(args.out_root).expanduser().resolve()
    out_images = out_root / "images"
    out_labels = out_root / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

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
        log.info(
            f"[modalities] {modalities} | energy_modality={energy_modality} | "
            f"num_slices={args.num_slices}"
        )

        rng_split = random.Random(args.seed)
        patient_dirs_ordered = list(patient_dirs)
        if not args.no_shuffle:
            rng_split.shuffle(patient_dirs_ordered)

        n_val = max(1, int(round(len(patient_dirs_ordered) * args.val_ratio)))
        if n_val >= len(patient_dirs_ordered):
            n_val = max(1, len(patient_dirs_ordered) - 1)
        eval_dirs = {p.name for p in patient_dirs_ordered[:n_val]}
        train_dirs = {p.name for p in patient_dirs_ordered[n_val:]}

        log.info(
            f"[split] train_patients={len(train_dirs)} eval_patients={len(eval_dirs)} "
            f"val_ratio={args.val_ratio:.2f}"
        )

        jobs = [
            (
                str(p),
                list(modalities),
                str(out_images),
                str(out_labels),
                int(args.num_slices),
                energy_modality,
            )
            for p in patient_dirs_ordered
        ]

        results: dict[str, Tuple[int, Optional[str]]] = {}
        max_workers = max(1, int(args.max_workers))

        if max_workers == 1:
            for job in jobs:
                pid, n_written, err = _process_patient_job(job)
                results[pid] = (n_written, err)
                if err:
                    log.warning(f"  {pid}: ERROR {err}")
                else:
                    log.info(f"  {pid}: wrote {n_written} files")
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_process_patient_job, job): job[0] for job in jobs}
                for fut in as_completed(futures):
                    pid, n_written, err = fut.result()
                    results[pid] = (n_written, err)
                    if err:
                        log.warning(f"  {pid}: ERROR {err}")
                    else:
                        log.info(f"  {pid}: wrote {n_written} files")

        train_tokens: List[str] = []
        eval_tokens: List[str] = []
        total_written = 0
        errors = 0
        for p in patient_dirs_ordered:
            n_written, err = results.get(p.name, (0, "missing result"))
            total_written += n_written
            if err:
                errors += 1
                continue
            if p.name in eval_dirs:
                eval_tokens.append(p.name)
            else:
                train_tokens.append(p.name)

        train_list_path = out_root / "train_list.txt"
        eval_list_path = out_root / "eval_list.txt"
        train_list_path.write_text("\n".join(sorted(train_tokens)) + "\n", encoding="utf-8")
        eval_list_path.write_text("\n".join(sorted(eval_tokens)) + "\n", encoding="utf-8")

        log.info(f"\n[done] total image/label files written: {total_written}")
        log.info(f"[done] patients with errors: {errors}")
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
