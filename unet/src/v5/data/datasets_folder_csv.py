from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_base_transform, UnsharpMask


class FolderUNetDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_size: int = 224,
        validate_images: bool = False,
        apply_unsharp: bool = False,
        unsharp_kernel_size: int = 5,
        unsharp_sigma: float = 1.0,
        unsharp_amount: float = 1.0,
        mindset_label_map_idx_1: str = "", 
        mindset_label_map_idx_2: str = "",
    ):
        self.csv_path = Path(csv_path)
        self.root = self.csv_path.parent
        self.df = pd.read_csv(self.csv_path)

        if "img_path" not in self.df.columns or "abnormal_type" not in self.df.columns:
            raise ValueError("CSV must contain columns: img_path, abnormal_type")

        self.transform = build_base_transform(image_size)

        self.apply_unsharp = apply_unsharp
        self.unsharp = UnsharpMask(unsharp_kernel_size, unsharp_sigma, unsharp_amount) if apply_unsharp else None

        self.df = self._apply_label_mapping(self.df)

        if validate_images:
            self.df = self._validate_images(self.df)

        self.df = self.df.reset_index(drop=True)
        
        self.mindset_label_map_idx_1 = mindset_label_map_idx_1
        self.mindset_label_map_idx_2 = mindset_label_map_idx_2

    def _apply_label_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.label_cfg is None:
            df["label_1"] = -1
            df["label_2"] = -1
            return df

        df["label_1"] = df["abnormal_type"].map(self.mindset_label_map_idx_1)
        df["label_2"] = df["abnormal_type"].map(self.mindset_label_map_idx_2)
        df = df.dropna(subset=["label_1", "label_2"]).copy()
        df["label_1"] = df["label_1"].astype(int)
        df["label_2"] = df["label_2"].astype(int)
        return df

    def _validate_images(self, df: pd.DataFrame) -> pd.DataFrame:
        keep_rows = []
        for i, row in df.iterrows():
            p = self.root / str(row["img_path"])
            if not p.exists():
                continue
            try:
                img = Image.open(p)
                img.verify()
                keep_rows.append(i)
            except Exception:
                continue
        return df.loc[keep_rows].copy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = self.root / str(row["img_path"])

        pil = Image.open(img_path).convert("L")
        x_orig = self.transform(pil)
        x_proc = self.unsharp(x_orig) if self.apply_unsharp and self.unsharp is not None else x_orig

        return {
            "input": x_proc,
            "target": x_proc,
            "original": x_orig,
            "label_1": int(row.get("label_1", -1)),
            "label_2": int(row.get("label_2", -1)),
            "path": str(img_path),
        }
