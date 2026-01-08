from __future__ import annotations

from typing import Any, Dict, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import build_base_transform, UnsharpMask


class AlzheimerUNetDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Any,
        image_size: int = 224,
        apply_unsharp: bool = False,
        unsharp_kernel_size: int = 5,
        unsharp_sigma: float = 1.0,
        unsharp_amount: float = 1.0,
    ):
        self.ds = hf_dataset
        self.transform = build_base_transform(image_size)
        self.apply_unsharp = apply_unsharp
        self.unsharp = UnsharpMask(unsharp_kernel_size, unsharp_sigma, unsharp_amount) if apply_unsharp else None

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[idx]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        x_orig = self.transform(img)  # [1,H,W] in [0,1]
        x_proc = self.unsharp(x_orig) if self.apply_unsharp and self.unsharp is not None else x_orig

        return {
            "input": x_proc,
            "target": x_proc,
            "original": x_orig,
            "label": int(row.get("label", -1)),
        }
