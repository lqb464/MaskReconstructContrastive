import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# Suppress verbose SimpleITK warnings that do not affect processing
sitk.ProcessObject_SetGlobalWarningDisplay(False)

MASK_KEYWORDS = ("mask", "seg", "label")
MRI_KEYWORDS = ("t1", "img", "image", "mri")

INTEGER_PIXEL_IDS = {
    sitk.sitkUInt8,
    sitk.sitkInt8,
    sitk.sitkUInt16,
    sitk.sitkInt16,
    sitk.sitkUInt32,
    sitk.sitkInt32,
    sitk.sitkUInt64,
    sitk.sitkInt64,
}


def load_nifti(nifti_path: str, cast_float: bool = False, desired_orientation: str = "RAI") -> sitk.Image:
    """
    Read a NIfTI file, drop any 4th dimension, optionally cast to float32,
    and reorient to the desired orientation (default RAI).
    """
    img = sitk.ReadImage(str(nifti_path))

    if img.GetDimension() == 4:
        img = img[:, :, :, 0]

    if cast_float:
        img = sitk.Cast(img, sitk.sitkFloat32)

    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(desired_orientation)
    return orienter.Execute(img)


def resample_isotropic(img: sitk.Image, spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear) -> sitk.Image:
    """Resample image to isotropic spacing while preserving orientation and origin."""
    original_size = img.GetSize()
    original_spacing = img.GetSpacing()

    new_size = [
        int(round(original_size[i] * original_spacing[i] / spacing[i]))
        for i in range(3)
    ]

    return sitk.Resample(
        img,
        new_size,
        sitk.Transform(),
        interpolator,
        img.GetOrigin(),
        spacing,
        img.GetDirection(),
        0,
        img.GetPixelID(),
    )


def _few_unique_mask(path: Path, max_unique: int = 16) -> bool:
    """
    Heuristic to decide if a file is a mask:
    - integer pixel type OR
    - <= max_unique non-zero unique values
    """
    img = sitk.ReadImage(str(path))
    if img.GetPixelID() in INTEGER_PIXEL_IDS:
        return True

    arr = sitk.GetArrayFromImage(img)
    nonzero = arr[arr != 0]
    if nonzero.size == 0:
        return True

    return np.unique(nonzero).size <= max_unique


def find_mri_and_mask(subject_dir: Path):
    """
    Identify MRI and mask files inside a subject directory.
    - If exactly two NIfTI files exist: choose mask by integer / few-unique heuristic.
    - Otherwise, fall back to filename keywords.
    """
    nii_files = sorted(
        [p for p in subject_dir.iterdir() if p.suffix in (".nii", ".gz") and str(p).endswith((".nii", ".nii.gz"))]
    )

    if len(nii_files) < 2:
        raise ValueError(f"Expected at least two NIfTI files in {subject_dir}, found {len(nii_files)}.")

    if len(nii_files) == 2:
        mask_flags = [_few_unique_mask(p) for p in nii_files]
        if mask_flags.count(True) == 1:
            mask_path = nii_files[mask_flags.index(True)]
            mri_path = nii_files[1 - mask_flags.index(True)]
            return mri_path, mask_path
        raise ValueError(
            f"Ambiguous MRI/mask detection in {subject_dir}. Heuristic could not pick a unique mask file."
        )

    def _has_keyword(path: Path, keywords) -> bool:
        name = path.name.lower()
        return any(k in name for k in keywords)

    mask_candidates = [p for p in nii_files if _has_keyword(p, MASK_KEYWORDS)]
    mri_candidates = [p for p in nii_files if _has_keyword(p, MRI_KEYWORDS)]

    if len(mask_candidates) == 1 and len(mri_candidates) == 1 and mask_candidates[0] != mri_candidates[0]:
        return mri_candidates[0], mask_candidates[0]

    raise ValueError(
        f"Could not uniquely determine MRI and mask files in {subject_dir}. "
        f"Name files with mask/seg/label or ensure only the MRI and mask are present."
    )


def ensure_mask_on_mri_grid(mri_img: sitk.Image, mask_img: sitk.Image, atol: float = 1e-5) -> sitk.Image:
    """Resample mask onto MRI grid if size/spacing/origin/direction differ."""
    same_size = mri_img.GetSize() == mask_img.GetSize()
    same_spacing = np.allclose(mri_img.GetSpacing(), mask_img.GetSpacing(), atol=atol)
    same_origin = np.allclose(mri_img.GetOrigin(), mask_img.GetOrigin(), atol=atol)
    same_direction = np.allclose(mri_img.GetDirection(), mask_img.GetDirection(), atol=atol)

    if same_size and same_spacing and same_origin and same_direction:
        return mask_img

    return sitk.Resample(
        mask_img,
        mri_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask_img.GetPixelID(),
    )


def extract_brain_slices_axial(volume_np: np.ndarray, n_slices: int = 50):
    """
    Select axial slice indices using an energy curve over the z-axis.
    Returns (slice_indices, (start, end)).
    """
    z = volume_np.shape[0]
    energy = volume_np.reshape(z, -1).mean(axis=1)
    energy_smooth = gaussian_filter1d(energy, sigma=5)

    threshold = energy_smooth.min() + 0.3 * (energy_smooth.max() - energy_smooth.min())
    brain_mask = energy_smooth > threshold
    idx = np.where(brain_mask)[0]
    if len(idx) == 0:
        start, end = int(z * 0.25), int(z * 0.75)
    else:
        start, end = idx[0], idx[-1]

    slice_indices = np.linspace(start, end, n_slices, dtype=int)
    slice_indices = np.unique(np.clip(slice_indices, 0, z - 1))
    return slice_indices, (start, end)


def extract_brain_slices_coronal(volume_np: np.ndarray, n_slices: int = 50):
    """
    Select coronal slice indices using an energy curve over the y-axis.
    Returns (slice_indices, (start, end)).
    """
    h = volume_np.shape[1]
    energy = volume_np.mean(axis=(0, 2))
    energy_smooth = gaussian_filter1d(energy, sigma=5)

    threshold = energy_smooth.min() + 0.4 * (energy_smooth.max() - energy_smooth.min())
    brain_mask = energy_smooth > threshold
    idx = np.where(brain_mask)[0]
    if len(idx) == 0:
        start, end = int(h * 0.25), int(h * 0.75)
    else:
        start, end = idx[0], idx[-1]

    slice_indices = np.linspace(start, end, n_slices, dtype=int)
    slice_indices = np.unique(np.clip(slice_indices, 0, h - 1))
    return slice_indices, (start, end)


def _extract_slice(image: sitk.Image, plane: str, idx: int) -> sitk.Image:
    """Extract a 2D slice using SimpleITK Extract to preserve metadata."""
    size = list(image.GetSize())
    extractor = sitk.ExtractImageFilter()

    if plane == "axial":
        extractor.SetSize([size[0], size[1], 0])
        extractor.SetIndex([0, 0, int(idx)])
    elif plane == "coronal":
        extractor.SetSize([size[0], 0, size[2]])
        extractor.SetIndex([0, int(idx), 0])
    else:
        raise ValueError(f"Unknown plane '{plane}'. Expected 'axial' or 'coronal'.")

    return extractor.Execute(image)


def save_slices_nii(
    mri_img: sitk.Image,
    mask_img: sitk.Image,
    plane: str,
    slice_indices,
    subject_id: str,
    output_root: Path,
    compress: bool,
):
    ext = ".nii.gz" if compress else ".nii"
    out_dir = output_root / plane / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in slice_indices:
        mri_slice = _extract_slice(mri_img, plane, idx)
        mask_slice = sitk.Cast(_extract_slice(mask_img, plane, idx), sitk.sitkUInt8)

        sitk.WriteImage(
            mri_slice,
            str(out_dir / f"{subject_id}_{plane}_{int(idx):03d}{ext}"),
            useCompression=compress,
        )
        sitk.WriteImage(
            mask_slice,
            str(out_dir / f"{subject_id}_{plane}_{int(idx):03d}_mask{ext}"),
            useCompression=compress,
        )


def save_sample_slice(
    mri_img: sitk.Image,
    mask_img: sitk.Image,
    plane: str,
    idx: int,
    subject_id: str,
    output_root: Path,
    compress: bool,
):
    ext = ".nii.gz" if compress else ".nii"
    output_root.mkdir(parents=True, exist_ok=True)

    mri_slice = _extract_slice(mri_img, plane, idx)
    mask_slice = sitk.Cast(_extract_slice(mask_img, plane, idx), sitk.sitkUInt8)

    sitk.WriteImage(
        mri_slice,
        str(output_root / f"{subject_id}_{plane}_sample_mri{ext}"),
        useCompression=compress,
    )
    sitk.WriteImage(
        mask_slice,
        str(output_root / f"{subject_id}_{plane}_sample_mask{ext}"),
        useCompression=compress,
    )


def process_subject(
    subject_dir: str,
    output_dir: str,
    n_axial: int,
    n_coronal: int,
    resample_iso: bool,
    compress: bool,
):
    subject_path = Path(subject_dir)
    subject_id = subject_path.name

    try:
        mri_path, mask_path = find_mri_and_mask(subject_path)

        mri_img = load_nifti(mri_path, cast_float=True)
        mask_img = load_nifti(mask_path, cast_float=False)

        mask_img = ensure_mask_on_mri_grid(mri_img, mask_img)

        if resample_iso:
            mri_iso = resample_isotropic(mri_img, spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear)
            mask_iso = sitk.Resample(
                mask_img,
                mri_iso,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                mask_img.GetPixelID(),
            )
            mri_img, mask_img = mri_iso, mask_iso

        if mask_img.GetPixelID() not in INTEGER_PIXEL_IDS:
            mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)

        volume_np = sitk.GetArrayFromImage(mri_img)
        axial_indices, _ = extract_brain_slices_axial(volume_np, n_slices=n_axial)
        coronal_indices, _ = extract_brain_slices_coronal(volume_np, n_slices=n_coronal)

        if len(axial_indices) == 0 or len(coronal_indices) == 0:
            raise ValueError("Slice selection produced zero indices.")

        out_root = Path(output_dir)
        save_slices_nii(mri_img, mask_img, "axial", axial_indices, subject_id, out_root, compress)
        save_slices_nii(mri_img, mask_img, "coronal", coronal_indices, subject_id, out_root, compress)

        sample_dir = out_root / "sample"
        axial_sample = int(axial_indices[len(axial_indices) // 2])
        coronal_sample = int(coronal_indices[len(coronal_indices) // 2])
        save_sample_slice(mri_img, mask_img, "axial", axial_sample, subject_id, sample_dir, compress)
        save_sample_slice(mri_img, mask_img, "coronal", coronal_sample, subject_id, sample_dir, compress)

        return subject_id, None
    except Exception as exc:  # noqa: BLE001
        return subject_id, str(exc)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract axial and coronal 2D NIfTI slices from synthstrip-data-v1-5-2d."
    )
    parser.add_argument("--data_root", required=True, help="Root directory containing subject folders.")
    parser.add_argument("--output_dir", required=True, help="Where to write axial, coronal, and sample slices.")
    parser.add_argument("--n_slices", type=int, default=50, help="Number of axial slices to save.")
    parser.add_argument("--n_coronal", type=int, default=50, help="Number of coronal slices to save.")
    parser.add_argument("--max_workers", type=int, default=8, help="ProcessPoolExecutor worker count.")
    parser.add_argument(
        "--resample_iso",
        action="store_true",
        help="Resample MRI (linear) and mask (nearest) to 1x1x1 mm before slicing.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Save slices as .nii.gz when enabled (otherwise .nii).",
    )
    return parser.parse_args()


def main():
    """
    Example:
        python scripts/extract_synthstrip_2d_nii.py \\
            --data_root /data/synthstrip-data-v1-5-2d \\
            --output_dir /data/synthstrip-2d-slices \\
            --n_slices 50 --n_coronal 50 --resample_iso --compress
    """
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    subject_dirs = [p for p in data_root.iterdir() if p.is_dir()]

    # Allow the user to pass a single subject directory directly (contains NIfTI files, no subfolders)
    if not subject_dirs:
        nifti_files = list(data_root.glob("*.nii")) + list(data_root.glob("*.nii.gz"))
        if nifti_files:
            subject_dirs = [data_root]
        else:
            raise SystemExit(f"No subject folders or NIfTI files found in {data_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "axial").mkdir(exist_ok=True)
    (output_dir / "coronal").mkdir(exist_ok=True)
    (output_dir / "sample").mkdir(exist_ok=True)

    errors = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_subject,
                str(subj),
                str(output_dir),
                args.n_slices,
                args.n_coronal,
                args.resample_iso,
                args.compress,
            ): subj.name
            for subj in subject_dirs
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="subjects"):
            sid, err = fut.result()
            if err:
                errors.append((sid, err))

    if errors:
        print("Completed with errors:")
        for sid, err in errors:
            print(f"  {sid}: {err}")
    else:
        print("All subjects processed successfully.")


if __name__ == "__main__":
    main()
