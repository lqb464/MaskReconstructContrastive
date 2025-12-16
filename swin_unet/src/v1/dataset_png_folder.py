# =============================================
# File: dataset_png_folder.py
# Read PNG images from folder/subfolders
# Phase 1: single-slice 2D MRI
# =============================================

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PngFolderDataset(Dataset):

    IMG_EXT = (".png", ".jpg", ".jpeg")

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 256,
        label: int = 0,
    ):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")

        self.image_size = image_size
        self.label = int(label)

        # collect all image paths
        self.image_paths: List[Path] = []
        for p in self.root_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.IMG_EXT:
                self.image_paths.append(p)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found under {self.root_dir}")

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),  # [1,H,W] in [0,1]
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]

        try:
            img = Image.open(img_path).convert("L")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {img_path}") from e

        x = self.transform(img)

        return {
            "image": x,
            "label": torch.tensor(self.label, dtype=torch.long),
        }
        