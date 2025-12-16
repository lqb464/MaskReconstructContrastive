from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image
import torch
from torch.utils.data import Dataset

from .transforms import build_base_transform, UnsharpMask
from .orientation import ensure_vertical_orientation


ORIENT_TO_AXIS = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2,
}


def _percentile_scale_01(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(x, p_low), np.percentile(x, p_high)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


@dataclass
class _VolumeCache:
    path: Optional[Path] = None
    vol01: Optional[np.ndarray] = None


class AdniNiftiSliceDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        image_size: int = 224,
        adni_image_type: str = "axial",
        adni_series_filter: Optional[str] = None,
        adni_label_csv: Optional[str] = None,
        middle_frac: float = 0.4,
        middle_subsample: int = 1,
        apply_unsharp: bool = False,
        unsharp_kernel_size: int = 5,
        unsharp_sigma: float = 1.0,
        unsharp_amount: float = 1.0,
    ):
        self.root = Path(root_dir)
        self.image_size = image_size
        self.axis = ORIENT_TO_AXIS.get(adni_image_type, 2)
        self.series_filter = adni_series_filter
        self.transform = build_base_transform(image_size)

        self.apply_unsharp = apply_unsharp
        self.unsharp = UnsharpMask(unsharp_kernel_size, unsharp_sigma, unsharp_amount) if apply_unsharp else None

        self.label_map = self._load_label_map(adni_label_csv) if adni_label_csv else {}
        self.index: List[Tuple[Path, int]] = []
        self.cache = _VolumeCache()

        self._build_index(middle_frac, middle_subsample)

    def _load_label_map(self, csv_path: str) -> Dict[str, int]:
        import csv

        label_map: Dict[str, int] = {}
        p = Path(csv_path)
        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = row["filename"]
                base = fn
                if base.endswith(".nii.gz"):
                    base = base[:-7]
                elif base.endswith(".nii"):
                    base = base[:-4]
                label_map[base] = int(row["label"])
        return label_map

    def _iter_nifti_files(self) -> List[Path]:
        files = sorted(list(self.root.rglob("*.nii")) + list(self.root.rglob("*.nii.gz")))
        if self.series_filter:
            files = [p for p in files if self.series_filter in p.name]
        return files

    def _build_index(self, middle_frac: float, middle_subsample: int) -> None:
        nifti_files = self._iter_nifti_files()
        for p in nifti_files:
            try:
                img = nib.load(str(p))
                shape = img.shape
                depth = shape[self.axis]
                center = depth // 2
                half_band = int(depth * middle_frac / 2.0)
                start = max(center - half_band, 0)
                stop = min(center + half_band, depth - 1)

                for sidx in range(start, stop + 1, max(1, int(middle_subsample))):
                    self.index.append((p, int(sidx)))
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.index)

    def _load_volume_norm01(self, path: Path) -> np.ndarray:
        if self.cache.path == path and self.cache.vol01 is not None:
            return self.cache.vol01

        img = nib.load(str(path))
        vol = img.get_fdata(dtype=np.float32)
        vol01 = _percentile_scale_01(vol)
        self.cache = _VolumeCache(path=path, vol01=vol01)
        return vol01

    def _extract_slice(self, vol01: np.ndarray, slice_idx: int) -> np.ndarray:
        if self.axis == 0:
            sl = vol01[slice_idx, :, :]
        elif self.axis == 1:
            sl = vol01[:, slice_idx, :]
        else:
            sl = vol01[:, :, slice_idx]
        sl = _percentile_scale_01(sl)
        sl = ensure_vertical_orientation(sl)
        return sl

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, sidx = self.index[idx]
        vol01 = self._load_volume_norm01(path)
        sl = self._extract_slice(vol01, sidx)

        img_u8 = (np.clip(sl, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(img_u8)
        x_orig = self.transform(pil)
        x_proc = self.unsharp(x_orig) if self.apply_unsharp and self.unsharp is not None else x_orig

        base = path.name
        if base.endswith(".nii.gz"):
            base = base[:-7]
        elif base.endswith(".nii"):
            base = base[:-4]
        label = int(self.label_map.get(base, -1))

        return {
            "input": x_proc,
            "target": x_proc,
            "original": x_orig,
            "label": label,
            "path": str(path),
            "slice_idx": int(sidx),
        }
