from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict

import torch
from torch.utils.data import Dataset

from data.dataset import plane_to_one_hot
from .io import load_png_grayscale, load_mask_npz

log = logging.getLogger(__name__)


class MaskReconstructionDataset(Dataset):
    """
    Dataset for PNG -> mask reconstruction.

    Expects a single folder containing:
      - images: *.png (default extension configurable)
      - masks:  *_mask.npz
    Pairing rule: for name.png, mask is name_mask.npz in the same folder.

    __getitem__ returns a dict with:
      - "input":  float tensor [1,H,W] in [0,1]
      - "target": float tensor [1,H,W] in {0,1}
      - "path":   str (image path)
      - "plane_one_hot": float tensor [2] (axial by default)
    """

    def __init__(
        self,
        data_dir: str | Path,
        image_ext: str = ".png",
        strict_pairs: bool = True,
        mask_key: Optional[str] = None,
        augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.data_dir = Path(data_dir).expanduser()
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")

        self.image_ext = image_ext.lower()
        self.strict_pairs = bool(strict_pairs)
        self.mask_key = mask_key
        self.augment = augment

        self.plane_one_hot = plane_to_one_hot("axial")  # default plane for all slices

        self.pairs: List[Tuple[Path, Path]] = []
        missing: List[Path] = []

        all_imgs = sorted(p for p in self.data_dir.iterdir() if p.is_file() and p.suffix.lower() == self.image_ext)
        for img_path in all_imgs:
            mask_path = img_path.with_name(f"{img_path.stem}_mask.npz")
            if mask_path.exists():
                self.pairs.append((img_path, mask_path))
            else:
                missing.append(img_path)

        if self.strict_pairs:
            if missing:
                sample = ", ".join(str(p.name) for p in missing[:5])
                raise FileNotFoundError(
                    f"Missing masks for {len(missing)} images (e.g., {sample}). "
                    f"Expected '*_mask.npz' next to each {self.image_ext} file."
                )
        else:
            if missing:
                log.warning(
                    "Skipping %d images without masks (strict_pairs=False). First few: %s",
                    len(missing),
                    ", ".join(str(p.name) for p in missing[:5]),
                )

        if len(self.pairs) == 0:
            raise RuntimeError(f"No image/mask pairs found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        img_path, mask_path = self.pairs[idx]
        x = load_png_grayscale(img_path)  # [1,H,W]
        y = load_mask_npz(mask_path, key=self.mask_key)  # [1,H,W]

        if self.augment is not None:
            x = self.augment(x)

        return {
            "input": x,
            "target": y,
            "path": str(img_path),
            "plane_one_hot": self.plane_one_hot.clone(),
        }


__all__ = ["MaskReconstructionDataset"]
