from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..data.pair_transforms import load_image_pil as _shared_load_image_pil

log = logging.getLogger(__name__)


def load_image_pil(path: str | Path) -> Image.Image:
    return _shared_load_image_pil(path)


def load_mask_pil_from_array(arr: np.ndarray) -> Image.Image:
    """
    Create PIL mask while preserving integer ids.
    Uses:
    - L (uint8) for ids <= 255
    - I;16 (uint16) for ids <= 65535
    - I (int32) otherwise
    """
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected mask shape {arr.shape}; expected [H,W].")

    if not np.issubdtype(arr.dtype, np.integer):
        arr = np.rint(arr).astype(np.int64)
    else:
        arr = arr.astype(np.int64, copy=False)

    max_id = int(arr.max()) if arr.size > 0 else 0
    min_id = int(arr.min()) if arr.size > 0 else 0
    if min_id >= 0 and max_id <= 255:
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    if min_id >= 0 and max_id <= 65535:
        return Image.fromarray(arr.astype(np.uint16), mode="I;16")
    if max_id > 255:
        log.warning("Mask id range exceeds uint8 (max=%d). Using 32-bit PIL mode.", max_id)
    return Image.fromarray(arr.astype(np.int32), mode="I")


def _ensure_mask_hw(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 2:
        pass
    elif mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    else:
        raise ValueError(f"Unexpected mask shape {mask.shape}; expected [H,W], [1,H,W], or [H,W,1].")
    if not np.issubdtype(mask.dtype, np.integer):
        mask = np.rint(mask).astype(np.int64)
    else:
        mask = mask.astype(np.int64, copy=False)
    return mask


def _resize_pair_tensors(img: torch.Tensor, mask: torch.Tensor, size: int, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    img, mask: [1,1,H,W]
    Returns resized tensors [1,1,size,size].
    """
    # Keep mask interpolation in float32 to avoid float64 promotion overhead on CPU workers.
    mask = mask.to(dtype=torch.float32)
    if mode == "direct":
        img_r = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
        mask_r = F.interpolate(mask, size=(size, size), mode="nearest")
        return img_r, mask_r

    if mode != "letterbox":
        raise ValueError(f"Unsupported resize_mode={mode}. Expected 'direct' or 'letterbox'.")

    h, w = int(img.shape[-2]), int(img.shape[-1])
    scale = min(size / float(w), size / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img_r = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
    mask_r = F.interpolate(mask, size=(new_h, new_w), mode="nearest")

    img_canvas = torch.zeros((1, 1, size, size), dtype=img.dtype, device=img.device)
    mask_canvas = torch.zeros((1, 1, size, size), dtype=mask.dtype, device=mask.device)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    img_canvas[:, :, top : top + new_h, left : left + new_w] = img_r
    mask_canvas[:, :, top : top + new_h, left : left + new_w] = mask_r
    return img_canvas, mask_canvas


def apply_pair_transforms(
    img_pil: Image.Image,
    mask_array: np.ndarray,
    image_size: int,
    do_hflip: bool = False,
    resize_mode: str = "direct",
) -> Tuple[torch.Tensor, torch.Tensor]:
    img_np = np.asarray(img_pil, dtype=np.float32) / 255.0
    mask_np = _ensure_mask_hw(mask_array)

    img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)

    if image_size > 0:
        img_t, mask_t = _resize_pair_tensors(img_t, mask_t, int(image_size), mode=resize_mode)

    if do_hflip:
        img_t = torch.flip(img_t, dims=[-1])
        mask_t = torch.flip(mask_t, dims=[-1])

    img_out = img_t.squeeze(0).clamp(0.0, 1.0)
    mask_out = mask_t.squeeze(0).round().to(dtype=torch.int64)
    assert img_out.shape[-2:] == mask_out.shape[-2:], "Image and mask shapes differ after transforms"
    return img_out, mask_out


__all__ = [
    "load_image_pil",
    "load_mask_pil_from_array",
    "apply_pair_transforms",
]
