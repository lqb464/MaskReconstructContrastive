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

def run_extract_IXI(data_dir, output_dir):
    max_workers = 10
    n_slices = 50

    nii_paths = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('.nii') or f.endswith('.nii.gz')
    ]

    errors = run_parallel_processing(nii_paths, output_dir, max_workers=max_workers, n_slices=n_slices)

    if errors:
        print("Errors encountered in the following subjects:")
        for sid, err in errors:
            print(f"{sid}: {err}")
    else:
        print("All images processed successfully.")

def run_extract_all_IXI():
    run_extract_IXI("E:/Data/IXI/IXI-T1", "data/IXI-T1")
    run_extract_IXI("E:/Data/IXI/IXI-T2", "data/IXI-T2")
    run_extract_IXI("E:/Data/IXI/IXI-PD", "data/IXI-PD")


if __name__ == '__main__':
    run_extract_all_IXI()

    # # test single file extraction
    # nii_input_dir = "E:/Data/IXI/IXI-PD/IXI002-Guys-0828-PD.nii.gz"
    #
    # itk_image = load_nifti(nii_input_dir)
    # itk_image = resample_isotropic(itk_image, spacing=(1.0, 1.0, 1.0))
    #
    # volume_np = sitk.GetArrayFromImage(itk_image)
    #
    # original_slices, slice_indices, brain_region, neck_slice_indices, neck_region = extract_brain_slices_axial(volume_np, n_slices=50)
    # original_slices, slice_indices, brain_region = extract_brain_slices_coronal(volume_np, n_slices=50)
    #
    # print(f"Extracted slices indices: {slice_indices}")
    # print(f"Brain region slice range: {brain_region}")
    #
    # save_slices_png(original_slices, "IXI002-Guys-0828", "data")