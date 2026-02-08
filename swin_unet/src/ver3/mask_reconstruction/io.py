from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


def load_png_grayscale(path: str | Path) -> torch.Tensor:
    """
    Legacy helper retained for compatibility.
    Load a PNG image as float32 tensor in [0,1] with shape [1,H,W].
    No resizing is applied.
    """
    p = Path(path)
    with Image.open(p) as img:
        if img.mode != "L":
            img = img.convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
    return tensor.clamp(0.0, 1.0)


def _first_npz_array(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    if len(npz.files) == 0:
        raise ValueError("NPZ file contains no arrays.")
    first_key = npz.files[0]
    return npz[first_key]


def load_mask_npz_array(path: str | Path, key: Optional[str] = None) -> np.ndarray:
    """
    Load mask from NPZ as integer numpy array [H,W] without binarization.
    """
    p = Path(path)
    with np.load(p) as data:
        arr = data[key] if key is not None else _first_npz_array(data)

    arr = np.asarray(arr)
    if arr.ndim == 2:
        pass
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Unexpected mask shape {arr.shape}; expected [H,W] or [1,H,W] or [H,W,1].")
    else:
        raise ValueError(f"Unexpected mask ndim={arr.ndim}; expected 2 or 3.")

    if not np.issubdtype(arr.dtype, np.integer):
        arr = np.rint(arr).astype(np.int64)
    else:
        arr = arr.astype(np.int64, copy=False)
    return arr


def load_mask_npz(path: str | Path, key: Optional[str] = None) -> torch.Tensor:
    """
    Legacy helper retained for compatibility with older callers.
    Current mask reconstruction dataset path uses load_mask_npz_array() + pair_transforms.

    Load mask from NPZ file.
    - Uses the provided key when given, otherwise the first array in the NPZ.
    - Returns float32 tensor [1,H,W] with values {0,1}.
    """
    arr = load_mask_npz_array(path, key=key)
    arr_bin = (arr != 0).astype(np.float32)[None, ...]
    tensor = torch.from_numpy(arr_bin)
    return tensor


__all__ = ["load_png_grayscale", "load_mask_npz", "load_mask_npz_array"]
