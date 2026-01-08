from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_base_transform, UnsharpMask


class AdniPrecomputedSliceDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        image_size: int = 224,
        meta_csv: str = "meta.csv",
        images_subdir: str = "images",
        apply_unsharp: bool = False,
        unsharp_kernel_size: int = 5,
        unsharp_sigma: float = 1.0,
        unsharp_amount: float = 1.0,
    ):
        self.root = Path(root_dir)
        self.images_dir = self.root / images_subdir
        self.df = pd.read_csv(self.root / meta_csv)
        self.transform = build_base_transform(image_size)

        self.apply_unsharp = apply_unsharp
        self.unsharp = UnsharpMask(unsharp_kernel_size, unsharp_sigma, unsharp_amount) if apply_unsharp else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = self.images_dir / str(row["img_path"])
        pil = Image.open(img_path).convert("L")
        x_orig = self.transform(pil)
        x_proc = self.unsharp(x_orig) if self.apply_unsharp and self.unsharp is not None else x_orig

        label = int(row["label"]) if "label" in row else -1
        slice_idx = int(row["slice_idx"]) if "slice_idx" in row else -1

        return {
            "input": x_proc,
            "target": x_proc,
            "original": x_orig,
            "label": label,
            "path": str(img_path),
            "slice_idx": slice_idx,
        }
