import argparse
import os

import SimpleITK as sitk
import imageio
import numpy as np
from scipy.ndimage import gaussian_filter1d

sitk.ProcessObject_SetGlobalWarningDisplay(False)


SUPPORTED_DIRECTIONS = {"axial", "coronal"}


def orient_to_rai(itk_image):
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("RAI")
    return orienter.Execute(itk_image)


def load_nifti(nifti_path):
    img = sitk.ReadImage(nifti_path)

    if img.GetDimension() == 4:
        img = img[:, :, :, 0]

    return orient_to_rai(img)


def build_reference_image(itk_image, spacing=(1.0, 1.0, 1.0)):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    new_size = [
        int(round(original_size[i] * original_spacing[i] / spacing[i]))
        for i in range(3)
    ]

    reference = sitk.Image(new_size, itk_image.GetPixelID())
    reference.SetSpacing(spacing)
    reference.SetOrigin(itk_image.GetOrigin())
    reference.SetDirection(itk_image.GetDirection())

    return reference


def resample_with_reference(itk_image, reference, interpolator):
    return sitk.Resample(
        itk_image,
        reference,
        sitk.Transform(),
        interpolator,
        0,
        itk_image.GetPixelID(),
    )


def extract_axial_indices(volume_np, n_slices=50):
    z_dim = volume_np.shape[0]

    energy = volume_np.reshape(z_dim, -1).mean(axis=1)
    energy_smooth = gaussian_filter1d(energy, sigma=5)

    threshold = energy_smooth.min() + 0.3 * (energy_smooth.max() - energy_smooth.min())
    brain_mask = energy_smooth > threshold

    idx = np.where(brain_mask)[0]
    if len(idx) == 0:
        start, end = int(z_dim * 0.25), int(z_dim * 0.65)
    else:
        start, end = idx[0], (idx[-1] - idx[0]) // 2 + idx[0]

    return np.linspace(start, end, n_slices, dtype=int)


def extract_coronal_indices(volume_np, n_slices=50):
    h_dim = volume_np.shape[1]

    energy = volume_np.mean(axis=(0, 2))
    energy_smooth = gaussian_filter1d(energy, sigma=5)

    threshold = energy_smooth.min() + 0.4 * (energy_smooth.max() - energy_smooth.min())
    brain_mask = energy_smooth > threshold

    idx = np.where(brain_mask)[0]
    if len(idx) == 0:
        start, end = int(h_dim * 0.25), int(h_dim * 0.75)
    else:
        start, end = idx[0], idx[-1]

    return np.linspace(start, end, n_slices, dtype=int)


def select_slice_indices(volume_np, direction, n_slices):
    if direction == "axial":
        return extract_axial_indices(volume_np, n_slices)
    if direction == "coronal":
        return extract_coronal_indices(volume_np, n_slices)
    raise ValueError(f"Unsupported direction: {direction}")


def normalize_image_slices(image_slices):
    img_min = min(s.min() for s in image_slices)
    img_max = max(s.max() for s in image_slices)
    denom = (img_max - img_min) + 1e-5

    normalized = []
    for s in image_slices:
        img = (s.astype(np.float32) - img_min) / denom
        normalized.append((img * 255).astype(np.uint8))

    return normalized


def prepare_mask_slices(mask_slices):
    prepared = []
    for s in mask_slices:
        mask_uint8 = (s > 0).astype(np.uint8) * 255
        prepared.append(mask_uint8)
    return prepared


def extract_slices(volume_np, indices, direction):
    if direction == "axial":
        return [volume_np[int(i), :, :] for i in indices]
    if direction == "coronal":
        return [volume_np[:, int(i), :] for i in indices]
    raise ValueError(f"Unsupported direction: {direction}")


def write_png_slices(image_slices, mask_slices, direction, output_dir, write_pairs=True):
    image_dir = os.path.join(output_dir, "image")
    mask_dir = os.path.join(output_dir, "mask")
    pairs_dir = os.path.join(output_dir, "pairs")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    if write_pairs:
        os.makedirs(pairs_dir, exist_ok=True)

    for i, (img, mask) in enumerate(zip(image_slices, mask_slices)):
        image_path = os.path.join(image_dir, f"{direction}_{i:03d}.png")
        mask_path = os.path.join(mask_dir, f"{direction}_{i:03d}.png")
        imageio.imwrite(image_path, img)
        imageio.imwrite(mask_path, mask)

        if write_pairs:
            if img.ndim == 2:
                combined = np.concatenate([img, mask], axis=1)
            else:
                combined = np.concatenate([img, mask], axis=1)
            pair_path = os.path.join(pairs_dir, f"{direction}_{i:03d}.png")
            imageio.imwrite(pair_path, combined)


def visualize_pairs(image_slices, mask_slices, direction, indices):
    import matplotlib.pyplot as plt

    for i, (img, mask, idx) in enumerate(zip(image_slices, mask_slices, indices)):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title(f"{direction} {int(idx)} image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title(f"{direction} {int(idx)} mask")
        axes[1].axis("off")

        fig.tight_layout()
        plt.show(block=True)
        plt.close(fig)


def find_required_files(input_dir):
    image_candidates = ["image.nii", "image.nii.gz"]
    mask_candidates = ["mask.nii", "mask.nii.gz"]

    image_path = None
    mask_path = None

    for name in image_candidates:
        candidate = os.path.join(input_dir, name)
        if os.path.isfile(candidate):
            image_path = candidate
            break

    for name in mask_candidates:
        candidate = os.path.join(input_dir, name)
        if os.path.isfile(candidate):
            mask_path = candidate
            break

    if image_path is None:
        raise FileNotFoundError("Missing image.nii or image.nii.gz in input directory")
    if mask_path is None:
        raise FileNotFoundError("Missing mask.nii or mask.nii.gz in input directory")

    return image_path, mask_path


def run_extract(input_dir, direction, output_dir, n_slices, visualize=False):
    if direction not in SUPPORTED_DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(SUPPORTED_DIRECTIONS)}")

    image_path, mask_path = find_required_files(input_dir)

    print("Found files:")
    print(f"  image: {image_path}")
    print(f"  mask:  {mask_path}")
    print(f"Direction: {direction}")
    print(f"n_slices: {n_slices}")
    print(f"Output dir: {output_dir}")

    image_itk = load_nifti(image_path)
    mask_itk = load_nifti(mask_path)

    reference = build_reference_image(image_itk, spacing=(1.0, 1.0, 1.0))

    image_resampled = resample_with_reference(image_itk, reference, sitk.sitkLinear)
    mask_resampled = resample_with_reference(mask_itk, reference, sitk.sitkNearestNeighbor)

    image_np = sitk.GetArrayFromImage(image_resampled)
    mask_np = sitk.GetArrayFromImage(mask_resampled)

    if image_np.shape != mask_np.shape:
        raise RuntimeError(
            f"Shape mismatch after resample: image {image_np.shape} vs mask {mask_np.shape}"
        )

    indices = select_slice_indices(image_np, direction, n_slices)

    image_slices = extract_slices(image_np, indices, direction)
    mask_slices = extract_slices(mask_np, indices, direction)

    image_slices_uint8 = normalize_image_slices(image_slices)
    mask_slices_uint8 = prepare_mask_slices(mask_slices)

    write_png_slices(image_slices_uint8, mask_slices_uint8, direction, output_dir, write_pairs=True)

    if visualize:
        visualize_pairs(image_slices_uint8, mask_slices_uint8, direction, indices)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Extract paired image and mask slices as PNG files."
    )
    parser.add_argument("--input-dir", required=True, help="Folder with image and mask NIfTI")
    parser.add_argument(
        "--direction",
        required=True,
        choices=sorted(SUPPORTED_DIRECTIONS),
        help="Slice direction",
    )
    parser.add_argument("--output-dir", required=True, help="Output folder")
    parser.add_argument(
        "--n-slices",
        type=int,
        default=50,
        help="Number of slices to extract",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize extracted pairs sequentially",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    run_extract(
        input_dir=args.input_dir,
        direction=args.direction,
        output_dir=args.output_dir,
        n_slices=args.n_slices,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    # Example usage:
    # python extract_image_mask_pairs.py --input-dir /path/subj1 --direction axial --output-dir out/subj1 --n-slices 50 --visualize
    # python extract_image_mask_pairs.py --input-dir /path/subj1 --direction coronal --output-dir out/subj1
    main()
