from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import shutil

import numpy as np
import torch
from torch.utils.data import Subset


def set_seed(seed: int) -> None:
    """Set python, numpy, torch RNG seeds."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cpu: bool) -> torch.device:
    """Return torch.device based on config."""
    if cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(p: Path) -> Path:
    """Create directory (parents=True) and return Path."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def has_labels_in_batch(batch: Dict) -> bool:
    """Return True if batch contains non-empty tensor under key 'label'."""
    y = batch.get("label", None)
    return isinstance(y, torch.Tensor) and y.numel() > 0


def extract_dataset_paths(dataset: Any) -> list[str]:
    """
    Extract sample paths from a dataset (including nested torch.utils.data.Subset).
    """
    if isinstance(dataset, Subset):
        base_paths = extract_dataset_paths(dataset.dataset)
        out = []
        for idx in dataset.indices:
            i = int(idx)
            if i < 0 or i >= len(base_paths):
                raise IndexError(
                    f"Subset index {i} is out of range for base dataset with {len(base_paths)} paths."
                )
            out.append(base_paths[i])
        return out

    if hasattr(dataset, "paths"):
        return [str(p) for p in getattr(dataset, "paths")]

    if hasattr(dataset, "pairs"):
        pairs = getattr(dataset, "pairs")
        out = []
        for pair in pairs:
            if isinstance(pair, (tuple, list)) and len(pair) >= 1:
                out.append(str(pair[0]))
            else:
                raise ValueError("Dataset pairs entry must be tuple/list with image path at index 0.")
        return out

    if hasattr(dataset, "images"):
        return [str(p) for p in getattr(dataset, "images")]

    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        raise TypeError("Unsupported dataset type for path extraction.")

    out = []
    for i in range(len(dataset)):
        item = dataset[i]
        if isinstance(item, dict) and ("path" in item):
            out.append(str(item["path"]))
        else:
            raise ValueError(
                "Dataset path extraction fallback expects dict samples with key 'path'. "
                f"Failed at index {i}."
            )
    return out


def write_path_list(paths: list[str], out_path: Path) -> Path:
    """Write one path per line and return output path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(p) for p in paths)
    if text:
        text += "\n"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def copy_images_to_dir(paths: list[str], out_dir: Path) -> int:
    """
    Copy image files to out_dir and return number of copied files.
    Files are prefixed with zero-padded index to avoid name collisions.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for i, p in enumerate(paths):
        src = Path(p)
        if not src.exists() or (not src.is_file()):
            continue
        dst_name = f"{i:06d}_{src.name}"
        dst = out_dir / dst_name
        shutil.copy2(src, dst)
        copied += 1
    return copied
