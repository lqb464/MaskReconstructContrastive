from cProfile import label
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import SimpleITK as sitk
import imageio
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

sitk.ProcessObject_SetGlobalWarningDisplay(False)

def load_dicom(dicom_dir):
    """
    dicom_dir = directory containing a single MRI series (all DICOM slices)
    returns: itk_image (SimpleITK)
    """

    # --- Load DICOM series ---
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_files)
    img = reader.Execute()

    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("RAI")
    img = orienter.Execute(img)

    return img

def load_nifti(nifti_path):
    """
    nifti_path = path to a .nii or .nii.gz file
    returns: itk_image (SimpleITK) in RAI orientation (same as load_dicom)
    """
    img = sitk.ReadImage(nifti_path)

    # If 4D, take the first volume (common for fMRI / DWI)
    if img.GetDimension() == 4:
        img = img[:, :, :, 0]

    # Cast to float32 for compatibility with downstream filters
    img = sitk.Cast(img, sitk.sitkFloat32)

    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("RAI")
    img = orienter.Execute(img)

    return img

def resample_isotropic(img, spacing=(1.0, 1.0, 1.0)):
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    new_size = [
        int(round(original_size[i] * original_spacing[i] / spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.Resample(
        img,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        img.GetOrigin(),
        spacing,
        img.GetDirection(),
        0,
        img.GetPixelID(),
    )
    return resampler

def extract_brain_slices_axial(volume_np, n_slices=50):
    """
    volume_np: numpy array of shape (Z, H, W)
    returns: list of 2D slices (numpy arrays)
    """

    Z = volume_np.shape[0]

    # --- 1. Compute energy (sum of intensities per slice) ---
    energy = volume_np.reshape(Z, -1).mean(axis=1)

    # --- 2. Smooth the curve to remove noise ---
    energy_smooth = gaussian_filter1d(energy, sigma=5)

    # --- 3. Determine brain region using threshold ---
    threshold = energy_smooth.min() + 0.3 * (energy_smooth.max() - energy_smooth.min())
    brain_mask = energy_smooth > threshold

    # Find continuous region
    idx = np.where(brain_mask)[0]
    if len(idx) == 0:
        # fallback: entire middle region
        start, end = int(Z * 0.25), int(Z * 0.65)
        neck_start, neck_end = end, Z
    else:
        start, end = idx[0], (idx[-1] - idx[0]) // 2 + idx[0]
        neck_start, neck_end = end, idx[-1]

    # --- 4. Select n evenly spaced slices from the brain region ---
    slice_indices = np.linspace(start, end, n_slices, dtype=int)
    # select n/2 slices from neck region
    neck_slice_indices = np.linspace(neck_start, neck_end, n_slices//2, dtype=int)

    slices = [volume_np[i] for i in slice_indices] + [volume_np[i] for i in neck_slice_indices]

    return slices, slice_indices, (start, end), neck_slice_indices, (neck_start, neck_end)


def extract_brain_slices_coronal(volume_np, n_slices=50):
    """
    volume_np: numpy array of shape (Z, H, W)
               Z = axial (inferior–superior)
               H = coronal (posterior–anterior)
               W = sagittal (left–right)

    returns:
        slices: list of 2D coronal slices (numpy arrays)
        slice_indices: indices along the coronal axis
        (start, end): selected coronal brain region
    """

    H = volume_np.shape[1]

    # --- 1. Compute energy per coronal slice ---
    # collapse Z and W
    energy = volume_np.mean(axis=(0, 2))

    # --- 2. Smooth the curve ---
    energy_smooth = gaussian_filter1d(energy, sigma=5)

    # --- 3. Determine brain region ---
    threshold = energy_smooth.min() + 0.4 * (energy_smooth.max() - energy_smooth.min())
    brain_mask = energy_smooth > threshold

    idx = np.where(brain_mask)[0]
    if len(idx) == 0:
        start, end = int(H * 0.25), int(H * 0.75)
    else:
        start, end = idx[0], idx[-1]

    # --- 4. Select evenly spaced coronal slices ---
    slice_indices = np.linspace(start, end, n_slices, dtype=int)

    # Coronal slices: fix H index, keep (Z, W)
    slices = [volume_np[:, i, :] for i in slice_indices]

    return slices, slice_indices, (start, end)

def save_slices_png(slices, sid, output_dir):
    img_max = max([s.max() for s in slices])
    img_min = min([s.min() for s in slices])

    for i, s in enumerate(slices):
        # --- Normalize to 0–255 for PNG ---
        img = s.astype(np.float32)
        img = img - img_min
        img = img / (img_max + 1e-5)
        img = (img * 255).astype(np.uint8)

        # --- File names ---
        path = os.path.join(output_dir, f"{sid}_{i:03d}.png")

        # --- Save PNG ---
        imageio.imwrite(path, img)

def process_image(nii_path, output_dir, n_slices=50):
    try:
        itk_image = load_nifti(nii_path)
        itk_image = resample_isotropic(itk_image, spacing=(1.0, 1.0, 1.0))
        volume_np = sitk.GetArrayFromImage(itk_image)

        sid = os.path.basename(nii_path).replace('.nii.gz', '').replace('.nii', '')

        axial_slices, _, _, _, _ = extract_brain_slices_axial(volume_np, n_slices=n_slices)
        save_slices_png(axial_slices, sid + "_axial", os.path.join(output_dir, "axial"))

        coronal_slices, _, _ = extract_brain_slices_coronal(volume_np, n_slices=n_slices)
        save_slices_png(coronal_slices, sid + "_coronal", os.path.join(output_dir, "coronal"))

        return (sid, None)
    except Exception as e:
        return (os.path.basename(nii_path), str(e))

def run_parallel_processing(nii_paths, output_dir, max_workers=10, n_slices=50):
    os.makedirs(os.path.join(output_dir, "axial"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "coronal"), exist_ok=True)

    errors = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_image, nii_path, output_dir, n_slices): nii_path
            for nii_path in nii_paths
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            sid, err = fut.result()
            if err:
                errors.append((sid, err))

    return errors


def _parse_case_id(case_id):
    parts = case_id.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid case folder name '{case_id}'. Expected <dataset>_<modality>_<subject_id>."
        )
    dataset = parts[0].strip()
    modality = parts[1].strip()
    if not dataset or not modality:
        raise ValueError(
            f"Invalid case folder name '{case_id}'. Dataset and modality must be non-empty."
        )
    return dataset, modality, (dataset, modality)


def _validate_case_folder(case_dir):
    entries = os.listdir(case_dir)
    image_candidates = []
    mask_candidates = []

    for name in entries:
        path = os.path.join(case_dir, name)
        if not os.path.isfile(path):
            continue
        if name in ("image.nii", "image.nii.gz"):
            image_candidates.append(name)
        elif name in ("mask.nii", "mask.nii.gz"):
            mask_candidates.append(name)

    if len(image_candidates) != 1:
        raise ValueError(
            f"Case folder '{os.path.basename(case_dir)}' must contain exactly one "
            f"image file named image.nii or image.nii.gz. Found: {image_candidates}"
        )
    if len(mask_candidates) != 1:
        raise ValueError(
            f"Case folder '{os.path.basename(case_dir)}' must contain exactly one "
            f"mask file named mask.nii or mask.nii.gz. Found: {mask_candidates}"
        )


def discover_cases(input_root):
    if not os.path.isdir(input_root):
        raise ValueError(f"Input root does not exist or is not a directory: {input_root}")

    cases = []
    for entry in sorted(os.listdir(input_root)):
        entry_path = os.path.join(input_root, entry)
        if not os.path.isdir(entry_path):
            continue
        _validate_case_folder(entry_path)
        cases.append(entry)

    return cases


def stratified_split_cases(cases, split_ratio=0.9, seed=42):
    groups = {}
    for case_id in cases:
        dataset, modality, strat_key = _parse_case_id(case_id)
        groups.setdefault(strat_key, []).append(case_id)

    rng = np.random.RandomState(seed)
    train_cases = []
    test_cases = []
    group_stats = {}
    warnings = []

    for strat_key in sorted(groups.keys()):
        group_cases = sorted(groups[strat_key])
        if not group_cases:
            continue

        perm = rng.permutation(len(group_cases))
        shuffled = [group_cases[i] for i in perm]

        n_total = len(shuffled)
        n_train = int(np.floor(n_total * split_ratio))
        n_test = n_total - n_train

        if n_test == 0 and n_total >= 2:
            n_test = 1
            n_train = n_total - 1

        if n_total >= 10 and n_test == 0:
            n_test = 1
            n_train = n_total - 1

        if n_train == 0 and n_total >= 1:
            n_train = 1
            n_test = n_total - 1

        group_train = shuffled[:n_train]
        group_test = shuffled[n_train:]

        train_cases.extend(group_train)
        test_cases.extend(group_test)

        group_stats[strat_key] = {
            "total": n_total,
            "train": len(group_train),
            "test": len(group_test),
        }

        if abs(len(group_train) - len(group_test)) > 1:
            dataset, modality = strat_key
            warnings.append(
                f"WARNING: Imbalanced split for {dataset}_{modality} "
                f"(train={len(group_train)}, test={len(group_test)})."
            )
        if len(group_test) == 0 and n_total >= 5:
            dataset, modality = strat_key
            warnings.append(
                f"WARNING: No test cases for {dataset}_{modality} (total={n_total})."
            )

    return train_cases, test_cases, group_stats, warnings


def _verify_split(cases, train_cases, test_cases):
    train_set = set(train_cases)
    test_set = set(test_cases)

    overlap = train_set.intersection(test_set)
    if overlap:
        raise ValueError(f"Split overlap detected: {sorted(list(overlap))}")

    if len(train_set) != len(train_cases) or len(test_set) != len(test_cases):
        raise ValueError("Duplicate case IDs detected in train or test splits.")

    if len(train_set) + len(test_set) != len(cases):
        raise ValueError(
            f"Split size mismatch. Total={len(cases)} "
            f"Train={len(train_cases)} Test={len(test_cases)}"
        )


def _build_split_summary(cases, train_cases, test_cases, group_stats):
    summary_groups = {}
    for (dataset, modality), stats in sorted(group_stats.items()):
        summary_groups[f"{dataset}_{modality}"] = {
            "total": stats["total"],
            "train": stats["train"],
            "test": stats["test"],
        }

    summary = {
        "total_cases": len(cases),
        "train_cases": len(train_cases),
        "test_cases": len(test_cases),
        "groups": summary_groups,
    }
    return summary


def _print_split_summary(total_cases, train_cases, test_cases, group_stats, warnings):
    print(f"Discovered cases: {total_cases}")
    print(f"Train/Test sizes: {len(train_cases)}/{len(test_cases)}")

    if group_stats:
        datasets = [k[0] for k in group_stats.keys()]
        modalities = [k[1] for k in group_stats.keys()]
        dataset_width = max(7, max(len(d) for d in datasets))
        modality_width = max(8, max(len(m) for m in modalities))

        header = (
            f"{'Dataset':<{dataset_width}}  "
            f"{'Modality':<{modality_width}}  "
            f"{'Total':>5}  {'Train':>5}  {'Test':>5}"
        )
        print(header)
        print("-" * len(header))

        for (dataset, modality), stats in sorted(group_stats.items()):
            print(
                f"{dataset:<{dataset_width}}  "
                f"{modality:<{modality_width}}  "
                f"{stats['total']:>5}  {stats['train']:>5}  {stats['test']:>5}"
            )

    for warning in warnings:
        print(warning)


def run_split_mode(input_root, output_root, split_ratio=0.9, seed=42, dry_run=False):
    cases = discover_cases(input_root)
    train_cases, test_cases, group_stats, warnings = stratified_split_cases(
        cases, split_ratio=split_ratio, seed=seed
    )

    _verify_split(cases, train_cases, test_cases)
    summary = _build_split_summary(cases, train_cases, test_cases, group_stats)
    _print_split_summary(len(cases), train_cases, test_cases, group_stats, warnings)

    if dry_run:
        print("Dry run enabled. No files were written.")
        return

    os.makedirs(output_root, exist_ok=True)

    train_path = os.path.join(output_root, "train_cases.txt")
    test_path = os.path.join(output_root, "test_cases.txt")
    summary_path = os.path.join(output_root, "split_summary.json")

    with open(train_path, "w") as f:
        for case_id in sorted(train_cases):
            f.write(f"{case_id}\n")

    with open(test_path, "w") as f:
        for case_id in sorted(test_cases):
            f.write(f"{case_id}\n")

    json = __import__("json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def _read_case_list(path):
    if not os.path.isfile(path):
        raise ValueError(f"Missing split file: {path}")
    case_ids = []
    with open(path, "r") as f:
        for line in f:
            case_id = line.strip()
            if case_id:
                case_ids.append(case_id)
    return case_ids


def _find_duplicate_case_ids(case_ids):
    seen = set()
    dupes = set()
    for case_id in case_ids:
        if case_id in seen:
            dupes.add(case_id)
        else:
            seen.add(case_id)
    return sorted(list(dupes))


def _validate_split_lists(train_cases, test_cases):
    train_dupes = _find_duplicate_case_ids(train_cases)
    if train_dupes:
        raise ValueError(f"Duplicate case IDs in train list: {train_dupes}")

    test_dupes = _find_duplicate_case_ids(test_cases)
    if test_dupes:
        raise ValueError(f"Duplicate case IDs in test list: {test_dupes}")

    overlap = sorted(list(set(train_cases).intersection(set(test_cases))))
    if overlap:
        raise ValueError(f"Overlap between train and test case lists: {overlap}")


def _ensure_case_folders_exist(input_root, case_ids):
    missing = []
    for case_id in case_ids:
        case_dir = os.path.join(input_root, case_id)
        if not os.path.isdir(case_dir):
            missing.append(case_id)
    if missing:
        raise ValueError(f"Missing case folders in input root: {missing}")


def _find_case_file(case_dir, base_name):
    candidates = []
    for ext in (".nii", ".nii.gz"):
        path = os.path.join(case_dir, f"{base_name}{ext}")
        if os.path.isfile(path):
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(
            f"Case folder '{os.path.basename(case_dir)}' must contain exactly one "
            f"{base_name} file named {base_name}.nii or {base_name}.nii.gz. Found: {candidates}"
        )
    return candidates[0]


def _prepare_mask_slice(mask_slice):
    if np.issubdtype(mask_slice.dtype, np.floating):
        if np.isnan(mask_slice).any():
            raise ValueError("Mask slice contains NaN values.")
        mask_slice = np.rint(mask_slice)

    mask_min = float(mask_slice.min())
    mask_max = float(mask_slice.max())

    if mask_min < 0:
        raise ValueError("Mask has negative values; cannot save to PNG without normalization.")
    if mask_max <= 255:
        return mask_slice.astype(np.uint8)
    if mask_max <= 65535:
        return mask_slice.astype(np.uint16)
    raise ValueError("Mask values exceed 65535; cannot save to PNG without normalization.")


def _save_slice_pairs(case_id, plane, image_slices, mask_slices, slice_indices, data_dir, label_dir):
    if len(slice_indices) == 0:
        raise ValueError(f"No slice indices for case '{case_id}' ({plane}).")

    if len(image_slices) != len(mask_slices):
        raise ValueError(
            f"Slice count mismatch for case '{case_id}' ({plane}). "
            f"Image={len(image_slices)} Mask={len(mask_slices)}"
        )

    if len(slice_indices) != len(image_slices):
        raise ValueError(
            f"Slice index count mismatch for case '{case_id}' ({plane}). "
            f"Indices={len(slice_indices)} Image={len(image_slices)}"
        )

    if len(set(slice_indices)) != len(slice_indices):
        raise ValueError(
            f"Output filename collision detected for case '{case_id}' ({plane}). "
            f"Duplicate slice indices: {sorted(list(slice_indices))}"
        )

    max_index = int(max(slice_indices))
    width = max(3, len(str(max_index)))

    img_max = max([s.max() for s in image_slices])
    img_min = min([s.min() for s in image_slices])

    for img_slice, mask_slice, slice_idx in zip(image_slices, mask_slices, slice_indices):
        filename = f"{case_id}_{plane}_{int(slice_idx):0{width}d}.png"
        image_path = os.path.join(data_dir, filename)
        label_path = os.path.join(label_dir, filename)

        img = img_slice.astype(np.float32)
        img = img - img_min
        img = img / (img_max + 1e-5)
        img = (img * 255).astype(np.uint8)
        imageio.imwrite(image_path, img)

        mask_out = _prepare_mask_slice(mask_slice)
        imageio.imwrite(label_path, mask_out)


def _process_case_extract(case_id, input_root, output_root, split_name, direction, n_slices):
    try:
        case_dir = os.path.join(input_root, case_id)
        image_path = _find_case_file(case_dir, "image")
        mask_path = _find_case_file(case_dir, "mask")

        image_itk = load_nifti(image_path)
        image_itk = resample_isotropic(image_itk, spacing=(1.0, 1.0, 1.0))
        image_np = sitk.GetArrayFromImage(image_itk)

        mask_itk = load_nifti(mask_path)
        mask_itk = resample_isotropic(mask_itk, spacing=(1.0, 1.0, 1.0))
        mask_np = sitk.GetArrayFromImage(mask_itk)

        if image_np.shape != mask_np.shape:
            raise ValueError(
                f"Image/mask shape mismatch for case '{case_id}'. "
                f"Image={image_np.shape} Mask={mask_np.shape}"
            )

        split_root = os.path.join(output_root, split_name)
        data_dir = os.path.join(split_root, "data")
        label_dir = os.path.join(split_root, "label")

        slices_written = 0

        if direction in ("axial", "both"):
            image_slices, slice_indices, _, neck_indices, _ = extract_brain_slices_axial(
                image_np, n_slices=n_slices
            )
            axial_indices = list(slice_indices) + list(neck_indices)
            mask_slices = [mask_np[i] for i in axial_indices]
            _save_slice_pairs(
                case_id,
                "axial",
                image_slices,
                mask_slices,
                axial_indices,
                data_dir,
                label_dir,
            )
            slices_written += len(axial_indices)

        if direction in ("coronal", "both"):
            image_slices, slice_indices, _ = extract_brain_slices_coronal(
                image_np, n_slices=n_slices
            )
            coronal_indices = list(slice_indices)
            mask_slices = [mask_np[:, i, :] for i in coronal_indices]
            _save_slice_pairs(
                case_id,
                "coronal",
                image_slices,
                mask_slices,
                coronal_indices,
                data_dir,
                label_dir,
            )
            slices_written += len(coronal_indices)

        return (case_id, slices_written, None)
    except Exception as e:
        return (case_id, 0, str(e))


def run_extract_mode(
    input_root,
    split_root,
    output_root,
    direction="axial",
    n_slices=50,
    max_workers=8,
):
    train_path = os.path.join(split_root, "train_cases.txt")
    test_path = os.path.join(split_root, "test_cases.txt")

    train_cases = _read_case_list(train_path)
    test_cases = _read_case_list(test_path)
    _validate_split_lists(train_cases, test_cases)

    _ensure_case_folders_exist(input_root, train_cases + test_cases)

    for split_name in ("train", "test"):
        os.makedirs(os.path.join(output_root, split_name, "data"), exist_ok=True)
        os.makedirs(os.path.join(output_root, split_name, "label"), exist_ok=True)

    split_map = {"train": train_cases, "test": test_cases}

    for split_name, case_ids in split_map.items():
        failures = []
        total_slices = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_case_extract,
                    case_id,
                    input_root,
                    output_root,
                    split_name,
                    direction,
                    n_slices,
                ): case_id
                for case_id in case_ids
            }

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Extracting {split_name}",
            ):
                case_id, slices_written, err = fut.result()
                if err:
                    failures.append((case_id, err))
                else:
                    total_slices += slices_written

        print(f"{split_name} cases processed: {len(case_ids)}")
        print(f"{split_name} total slices written: {total_slices}")
        if failures:
            print(f"{split_name} failed cases: {[c for c, _ in failures]}")
            for case_id, err in failures:
                print(f"ERROR: {case_id}: {err}")


if __name__ == '__main__':

    argparse = __import__("argparse")
    parser = argparse.ArgumentParser(description="MRI slice extraction and dataset splitting.")
    parser.add_argument("--mode", default=None, help="Execution mode. Use 'split' for dataset splitting.")
    parser.add_argument("--input-root", dest="input_root", default=None, help="Root folder of case subfolders.")
    parser.add_argument("--split-root", dest="split_root", default=None, help="Folder containing split files.")
    parser.add_argument("--output-root", dest="output_root", default=None, help="Output folder for split files.")
    parser.add_argument("--split-ratio", dest="split_ratio", type=float, default=0.9)
    parser.add_argument("--seed", dest="seed", type=int, default=42)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--direction", dest="direction", default="axial")
    parser.add_argument("--n-slices", dest="n_slices", type=int, default=50)
    parser.add_argument("--max-workers", dest="max_workers", type=int, default=8)

    args, _ = parser.parse_known_args()

    if args.mode == "split":
        if not args.input_root or not args.output_root:
            parser.error("--input-root and --output-root are required when --mode split is used.")
        run_split_mode(
            input_root=args.input_root,
            output_root=args.output_root,
            split_ratio=args.split_ratio,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        raise SystemExit(0)

    if args.mode == "extract":
        if not args.input_root or not args.output_root or not args.split_root:
            parser.error(
                "--input-root, --split-root, and --output-root are required when --mode extract is used."
            )
        if args.direction not in ("axial", "coronal", "both"):
            parser.error("--direction must be one of: axial, coronal, both.")
        run_extract_mode(
            input_root=args.input_root,
            split_root=args.split_root,
            output_root=args.output_root,
            direction=args.direction,
            n_slices=args.n_slices,
            max_workers=args.max_workers,
        )
        raise SystemExit(0)

    # test single file extraction
    nii_input_path = "data/sample/image.nii.gz"

    itk_image = load_nifti(nii_input_path)
    itk_image = resample_isotropic(itk_image, spacing=(1.0, 1.0, 1.0))

    volume_np = sitk.GetArrayFromImage(itk_image)

    # original_slices, slice_indices, brain_region, neck_slice_indices, neck_region = extract_brain_slices_axial(volume_np, n_slices=50)
    original_slices, slice_indices, brain_region = extract_brain_slices_coronal(volume_np, n_slices=50)

    print(f"Extracted slices indices: {slice_indices}")
    print(f"Brain region slice range: {brain_region}")

    label_input_path = "data/sample/mask.nii.gz"

    label_itk_image = load_nifti(label_input_path)
    label_itk_image = resample_isotropic(label_itk_image, spacing=(1.0, 1.0, 1.0))

    label_volume_np = sitk.GetArrayFromImage(label_itk_image)

    label_slices = [label_volume_np[:, i, :] for i in slice_indices]

    save_slices_png(original_slices, "sample_image_coronal", "data")
    save_slices_png(label_slices, "sample_label_coronal", "data")
