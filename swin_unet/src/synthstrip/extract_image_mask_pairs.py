import argparse
import os

import SimpleITK as sitk
import imageio
import numpy as np

from extract_MRI_2d_image import load_nifti, resample_isotropic

sitk.ProcessObject_SetGlobalWarningDisplay(False)

SUPPORTED_DIRECTIONS = {"axial", "coronal"}


def resample_isotropic_mask(itk_image, spacing=(1.0, 1.0, 1.0), reference=None):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    if reference is None:
        new_size = [
            int(round(original_size[i] * original_spacing[i] / spacing[i]))
            for i in range(3)
        ]
        reference = sitk.Image(new_size, itk_image.GetPixelID())
        reference.SetSpacing(spacing)
        reference.SetOrigin(itk_image.GetOrigin())
        reference.SetDirection(itk_image.GetDirection())

    resampled = sitk.Resample(
        itk_image,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        itk_image.GetPixelID(),
    )
    return resampled


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


def build_uniform_slice_indices(volume_np, direction, n_slices):
    if n_slices <= 0:
        raise ValueError("n_slices must be a positive integer")

    if direction == "axial":
        axis_len = volume_np.shape[0]
    elif direction == "coronal":
        axis_len = volume_np.shape[1]
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    if n_slices > axis_len:
        raise ValueError(
            f"n_slices ({n_slices}) exceeds available slices ({axis_len})"
        )

    return np.linspace(0, axis_len - 1, n_slices, dtype=int).tolist()


def build_image_slices(volume_np, indices, direction):
    if direction == "axial":
        return [volume_np[int(i), :, :] for i in indices]
    if direction == "coronal":
        return [volume_np[:, int(i), :] for i in indices]
    raise ValueError(f"Unsupported direction: {direction}")


def extract_mask_slices(volume_np, indices, direction):
    if direction == "axial":
        return [volume_np[int(i), :, :] for i in indices]
    if direction == "coronal":
        return [volume_np[:, int(i), :] for i in indices]
    raise ValueError(f"Unsupported direction: {direction}")


def normalize_image_slices(image_slices):
    normalized = []
    for s in image_slices:
        img = s.astype(np.float32)
        img_min = img.min()
        img_max = img.max()
        denom = (img_max - img_min) + 1e-5
        img = (img - img_min) / denom
        normalized.append((img * 255).astype(np.uint8))

    return normalized


def prepare_mask_slices(mask_slices):
    prepared = []
    for s in mask_slices:
        mask_uint8 = (s > 0).astype(np.uint8) * 255
        prepared.append(mask_uint8)
    return prepared


def save_png_slices(image_slices, mask_slices, direction, indices, output_dir):
    image_dir = os.path.join(output_dir, "image")
    mask_dir = os.path.join(output_dir, "mask")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for img, mask, idx in zip(image_slices, mask_slices, indices):
        image_path = os.path.join(image_dir, f"{direction}_{int(idx):03d}.png")
        mask_path = os.path.join(mask_dir, f"{direction}_{int(idx):03d}.png")
        imageio.imwrite(image_path, img)
        imageio.imwrite(mask_path, mask)


def visualize_pairs(image_slices, mask_slices, indices, direction):
    import matplotlib.pyplot as plt

    for img, mask, idx in zip(image_slices, mask_slices, indices):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title(f"{direction} {int(idx):03d} image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title(f"{direction} {int(idx):03d} mask")
        axes[1].axis("off")

        fig.tight_layout()
        plt.show(block=True)
        plt.close(fig)


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

    image_resampled = resample_isotropic(image_itk, spacing=(1.0, 1.0, 1.0))
    mask_resampled = resample_isotropic_mask(
        mask_itk, spacing=(1.0, 1.0, 1.0), reference=image_resampled
    )

    image_np = sitk.GetArrayFromImage(image_resampled)
    mask_np = sitk.GetArrayFromImage(mask_resampled)

    if image_np.shape != mask_np.shape:
        raise RuntimeError(
            f"Shape mismatch after resample: image {image_np.shape} vs mask {mask_np.shape}"
        )

    slice_indices = build_uniform_slice_indices(image_np, direction, n_slices)
    image_slices = build_image_slices(image_np, slice_indices, direction)
    mask_slices = extract_mask_slices(mask_np, slice_indices, direction)

    image_slices_uint8 = normalize_image_slices(image_slices)
    mask_slices_uint8 = prepare_mask_slices(mask_slices)

    save_png_slices(
        image_slices_uint8, mask_slices_uint8, direction, slice_indices, output_dir
    )

    if visualize:
        visualize_pairs(image_slices_uint8, mask_slices_uint8, slice_indices, direction)


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
