import argparse
import json
import os
import random

import imageio
import numpy as np

SUPPORTED_DIRECTIONS = {"axial", "coronal"}


def _read_grayscale(path):
    img = imageio.v2.imread(path)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = img.mean(axis=2)
    return img


def _parse_index(filename, direction):
    stem = os.path.splitext(filename)[0]
    prefix = f"{direction}_"
    if not stem.startswith(prefix):
        return None
    idx_str = stem[len(prefix):]
    if not idx_str.isdigit():
        return None
    return int(idx_str)


def _collect_pngs(folder, direction):
    if not os.path.isdir(folder):
        return None, [f"Missing folder: {folder}"]
    files = [f for f in os.listdir(folder) if f.endswith(".png")]
    matched = []
    errors = []
    for f in files:
        if not f.startswith(f"{direction}_"):
            continue
        idx = _parse_index(f, direction)
        if idx is None:
            errors.append(f"Unparseable filename: {f}")
            continue
        matched.append((idx, f))
    return matched, errors


def _seed_rng():
    seed_env = os.environ.get("SEED")
    if seed_env is not None:
        try:
            seed = int(seed_env)
        except ValueError:
            seed = 42
    else:
        seed = 42
    random.seed(seed)


def verify(output_dir, direction, max_samples, fail_fast, report_json):
    failures = []
    warnings = []
    metrics = {
        "output_dir": output_dir,
        "direction": direction,
        "image_count": 0,
        "mask_count": 0,
        "pair_count": 0,
        "non_binary_masks": [],
        "nearly_constant_images": [],
        "mask_nonzero_ratios": [],
        "mask_zero_count": 0,
        "top_mask_ratios": [],
        "sample_checks": [],
        "folder_checks": {},
        "filename_mismatches": {},
    }

    image_dir = os.path.join(output_dir, "image")
    mask_dir = os.path.join(output_dir, "mask")

    metrics["folder_checks"]["image_dir"] = os.path.isdir(image_dir)
    metrics["folder_checks"]["mask_dir"] = os.path.isdir(mask_dir)

    if not os.path.isdir(image_dir):
        failures.append(f"Missing folder: {image_dir}")
        if fail_fast:
            return failures, warnings, metrics
    if not os.path.isdir(mask_dir):
        failures.append(f"Missing folder: {mask_dir}")
        if fail_fast:
            return failures, warnings, metrics

    image_files, image_errors = _collect_pngs(image_dir, direction)
    mask_files, mask_errors = _collect_pngs(mask_dir, direction)

    if image_errors:
        failures.extend(image_errors)
        if fail_fast:
            return failures, warnings, metrics
    if mask_errors:
        failures.extend(mask_errors)
        if fail_fast:
            return failures, warnings, metrics

    if image_files is None or mask_files is None:
        return failures, warnings, metrics

    image_map = {idx: fname for idx, fname in image_files}
    mask_map = {idx: fname for idx, fname in mask_files}

    metrics["image_count"] = len(image_map)
    metrics["mask_count"] = len(mask_map)

    image_indices = set(image_map.keys())
    mask_indices = set(mask_map.keys())

    if image_indices != mask_indices:
        missing_in_mask = sorted(image_indices - mask_indices)
        missing_in_image = sorted(mask_indices - image_indices)
        metrics["filename_mismatches"]["missing_in_mask"] = missing_in_mask
        metrics["filename_mismatches"]["missing_in_image"] = missing_in_image
        failures.append("Image/mask filename sets do not match")
        if fail_fast:
            return failures, warnings, metrics

    common_indices = sorted(image_indices & mask_indices)
    metrics["pair_count"] = len(common_indices)

    if not common_indices:
        failures.append("No matching image/mask pairs found")
        if fail_fast:
            return failures, warnings, metrics

    image_stats = {}
    mask_ratios = []
    mask_zero_count = 0

    for idx in common_indices:
        img_path = os.path.join(image_dir, image_map[idx])
        mask_path = os.path.join(mask_dir, mask_map[idx])

        img = _read_grayscale(img_path)
        mask = _read_grayscale(mask_path)

        img = img.astype(np.float32)
        mask = mask.astype(np.float32)

        img_min = float(img.min())
        img_max = float(img.max())
        img_mean = float(img.mean())
        img_std = float(img.std())

        image_stats[idx] = {
            "min": img_min,
            "max": img_max,
            "mean": img_mean,
            "std": img_std,
        }

        if img_std < 1.0:
            metrics["nearly_constant_images"].append(image_map[idx])

        uniq = np.unique(mask)
        uniq_set = set(int(v) for v in uniq.tolist())
        if not uniq_set.issubset({0, 255}):
            metrics["non_binary_masks"].append(
                {"file": mask_map[idx], "unique_values": sorted(uniq_set)}
            )

        mask_bin = mask > 0
        ratio = float(mask_bin.mean())
        mask_ratios.append((idx, ratio))
        if ratio == 0.0:
            mask_zero_count += 1

    metrics["mask_zero_count"] = mask_zero_count
    metrics["mask_nonzero_ratios"] = [r for _, r in mask_ratios]

    if metrics["non_binary_masks"]:
        failures.append("Non-binary masks detected")
        if fail_fast:
            return failures, warnings, metrics

    if metrics["nearly_constant_images"]:
        warnings.append(
            f"Nearly-constant images (std < 1.0): {len(metrics['nearly_constant_images'])}"
        )

    ratios_only = [r for _, r in mask_ratios]
    metrics["mask_ratio_stats"] = {
        "min": float(np.min(ratios_only)),
        "median": float(np.median(ratios_only)),
        "max": float(np.max(ratios_only)),
    }

    top_k = sorted(mask_ratios, key=lambda x: x[1], reverse=True)[:5]
    metrics["top_mask_ratios"] = [
        {"file": image_map[idx], "mask_ratio": ratio} for idx, ratio in top_k
    ]

    _seed_rng()
    sample_indices = common_indices[:]
    random.shuffle(sample_indices)
    sample_indices = sample_indices[: max_samples]

    for idx in sample_indices:
        img_path = os.path.join(image_dir, image_map[idx])
        mask_path = os.path.join(mask_dir, mask_map[idx])

        img = _read_grayscale(img_path).astype(np.float32)
        mask = _read_grayscale(mask_path).astype(np.float32)
        mask_bin = mask > 0
        ratio = float(mask_bin.mean())

        entry = {
            "file": image_map[idx],
            "mask_ratio": ratio,
            "mean_in": None,
            "mean_out": None,
        }

        if ratio > 0.0:
            ys, xs = np.where(mask_bin)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            bbox_mask = mask_bin[y0 : y1 + 1, x0 : x1 + 1]
            bbox_img = img[y0 : y1 + 1, x0 : x1 + 1]

            inside_vals = bbox_img[bbox_mask]
            outside_vals = img[~mask_bin]

            entry["mean_in"] = float(inside_vals.mean()) if inside_vals.size else None
            entry["mean_out"] = float(outside_vals.mean()) if outside_vals.size else None

        metrics["sample_checks"].append(entry)

    if report_json:
        report = {
            "failures": failures,
            "warnings": warnings,
            "metrics": metrics,
        }
        os.makedirs(os.path.dirname(report_json) or ".", exist_ok=True)
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return failures, warnings, metrics


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Verify SynthStrip-2D extracted image/mask pairs."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder that contains image/ and mask/",
    )
    parser.add_argument(
        "--direction",
        required=True,
        choices=sorted(SUPPORTED_DIRECTIONS),
        help="Slice direction (axial or coronal)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Number of random pairs for deeper checks",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit non-zero on first failure",
    )
    parser.add_argument(
        "--report-json",
        help="Optional path to save a JSON report",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    failures, warnings, metrics = verify(
        output_dir=args.output_dir,
        direction=args.direction,
        max_samples=args.max_samples,
        fail_fast=args.fail_fast,
        report_json=args.report_json,
    )

    print("Verification summary")
    print(f"  output_dir: {args.output_dir}")
    print(f"  direction: {args.direction}")
    print(f"  images: {metrics.get('image_count', 0)}")
    print(f"  masks: {metrics.get('mask_count', 0)}")
    print(f"  pairs: {metrics.get('pair_count', 0)}")

    if "mask_ratio_stats" in metrics:
        stats = metrics["mask_ratio_stats"]
        print(
            "  mask_ratio (min/median/max): "
            f"{stats['min']:.4f} / {stats['median']:.4f} / {stats['max']:.4f}"
        )
        print(f"  empty masks: {metrics['mask_zero_count']}")

    if metrics.get("top_mask_ratios"):
        print("  top mask ratios:")
        for entry in metrics["top_mask_ratios"]:
            print(f"    {entry['file']}  ratio={entry['mask_ratio']:.4f}")

    if metrics.get("sample_checks"):
        print("  sample checks:")
        for entry in metrics["sample_checks"]:
            mean_in = entry["mean_in"]
            mean_out = entry["mean_out"]
            if mean_in is None or mean_out is None:
                line = (
                    f"    {entry['file']}  ratio={entry['mask_ratio']:.4f} "
                    f"mean_in=None mean_out=None"
                )
            else:
                line = (
                    f"    {entry['file']}  ratio={entry['mask_ratio']:.4f} "
                    f"mean_in={mean_in:.2f} mean_out={mean_out:.2f}"
                )
            print(line)

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    print("PASS")


if __name__ == "__main__":
    # Example:
    # python verify_extracted_pairs.py --output-dir out/subj1 --direction axial --max-samples 10 --report-json out/report.json
    main()
