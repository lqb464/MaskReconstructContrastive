from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..data.dataset import plane_to_one_hot
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
        mask_suffix: str = "_mask.npz",
        strict_pairs: bool = True,
        mask_key: Optional[str] = None,
        augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        image_size: Optional[int] = None,
        target_size: int = 0,
        resize_mode: str = "letterbox",
        debug_shapes: bool = False,
    ):
        self.data_dir = Path(data_dir).expanduser()
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")

        self.image_ext = image_ext.lower()
        self.mask_suffix = mask_suffix
        self.strict_pairs = bool(strict_pairs)
        self.mask_key = mask_key
        self.augment = augment
        self.image_size = image_size
        self.target_size = int(target_size)
        self.resize_mode = resize_mode
        self.debug_shapes = debug_shapes

        self.plane_one_hot = plane_to_one_hot("axial")  # default plane for all slices

        self.pairs: List[Tuple[Path, Path]] = []
        missing: List[Path] = []

        all_imgs = sorted(p for p in self.data_dir.iterdir() if p.is_file() and p.suffix.lower() == self.image_ext)
        for img_path in all_imgs:
            mask_path = img_path.with_name(f"{img_path.stem}{self.mask_suffix}")
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

        # initial shape check
        if x.shape[-2:] != y.shape[-2:]:
            if self.debug_shapes:
                print(f"[warn] shape mismatch before resize: {x.shape} vs {y.shape} for {img_path.name}")
            # fallback: resize mask to image size
            y = F.interpolate(y.unsqueeze(0), size=x.shape[-2:], mode="nearest").squeeze(0)

        # paired resize logic
        target_sz = self.target_size if self.target_size > 0 else self.image_size
        if target_sz is not None and target_sz > 0:
            x, y = resize_pair(x, y, target_sz, mode=self.resize_mode)

        if self.augment is not None:
            x = self.augment(x)

        assert x.shape[-2:] == y.shape[-2:], f"Shape mismatch after resize: {x.shape} vs {y.shape}"
        if self.debug_shapes and idx < 3:
            print(f"[debug] sample {idx}: shape {x.shape[-2:]} mode={self.resize_mode} target_sz={target_sz}")

        return {
            "input": x,
            "target": y,
            "path": str(img_path),
            "plane_one_hot": self.plane_one_hot.clone(),
        }


def resize_pair(x: torch.Tensor, y: torch.Tensor, target_size: int, mode: str = "letterbox") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resize image/mask pair consistently.
    x, y: [1,H,W]; y binary.
    """
    H, W = x.shape[-2:]
    if mode == "direct":
        x_r = F.interpolate(x.unsqueeze(0), size=(target_size, target_size), mode="bilinear", align_corners=False).squeeze(0)
        y_r = F.interpolate(y.unsqueeze(0), size=(target_size, target_size), mode="nearest").squeeze(0)
        y_r = (y_r > 0.5).float()
        return x_r, y_r

    # letterbox
    scale = min(target_size / H, target_size / W)
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))

    x_scaled = F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
    y_scaled = F.interpolate(y.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)
    y_scaled = (y_scaled > 0.5).float()

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    x_padded = F.pad(x_scaled, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    y_padded = F.pad(y_scaled, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    return x_padded, y_padded


__all__ = ["MaskReconstructionDataset", "resize_pair"]
