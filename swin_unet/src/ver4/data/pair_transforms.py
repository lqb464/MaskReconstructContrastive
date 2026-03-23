from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def load_image_pil(path: str | Path) -> Image.Image:
    img = Image.open(path).convert("L")
    return img

def load_mask_pil_from_array(arr: np.ndarray) -> Image.Image:
    """Create a PIL mask from numpy array, preserving integer ids."""
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    arr_uint8 = arr.astype(np.uint8)
    return Image.fromarray(arr_uint8, mode="L")

def resize_pair(img: Image.Image, mask: Image.Image, size: int, mode: str = "direct") -> Tuple[Image.Image, Image.Image]:
    """Resize image/mask to square size with appropriate interpolation."""
    if mode == "direct":
        img_r = img.resize((size, size), resample=Image.BILINEAR)
        mask_r = mask.resize((size, size), resample=Image.NEAREST)
        return img_r, mask_r

    w, h = img.size
    scale = min(size / float(w), size / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img_r = img.resize((new_w, new_h), resample=Image.BILINEAR)
    mask_r = mask.resize((new_w, new_h), resample=Image.NEAREST)

    canvas_img = Image.new("L", (size, size), color=0)
    canvas_mask = Image.new("L", (size, size), color=0)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas_img.paste(img_r, (left, top))
    canvas_mask.paste(mask_r, (left, top))
    return canvas_img, canvas_mask

def hflip_pair(img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Horizontal flip on tensors shaped [1,H,W]."""
    return torch.flip(img, dims=[2]), torch.flip(mask, dims=[2])

def to_tensor_gray01(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).clamp(0.0, 1.0)

def to_tensor_mask(mask: Image.Image) -> torch.Tensor:
    arr = np.array(mask, dtype=np.int64)
    return torch.from_numpy(arr).unsqueeze(0)

def apply_pair_transforms(img_pil: Image.Image, mask_pil: Image.Image, image_size: int, do_hflip: bool = False, resize_mode: str = "direct") -> Tuple[torch.Tensor, torch.Tensor]:
    img_r, mask_r = resize_pair(img_pil, mask_pil, image_size, mode=resize_mode)
    img_t = to_tensor_gray01(img_r)
    mask_t = to_tensor_mask(mask_r)
    if do_hflip:
        img_t, mask_t = hflip_pair(img_t, mask_t)
    assert img_t.shape[-2:] == mask_t.shape[-2:], "Image and mask shapes differ after transforms"
    return img_t, mask_t

__all__ = [
    "load_image_pil",
    "load_mask_pil_from_array",
    "resize_pair",
    "hflip_pair",
    "to_tensor_gray01",
    "to_tensor_mask",
    "apply_pair_transforms",
]
