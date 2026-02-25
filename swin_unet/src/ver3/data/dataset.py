
# =============================================
# File: data.py
# Dataset + DataLoader for folder-with-subfolders image dataset
# Phase 1: SwinUNet Dual View SSL (no preprocessing here)
# =============================================
from __future__ import annotations

import os
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    raise ImportError("PIL (Pillow) is required for image loading. Please install pillow.") from e


# -------------------------
# Helpers
# -------------------------
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in _IMG_EXTS


def _canonicalize_path(p: str | Path) -> str:
    """Canonical string path for stable matching (resolve + normcase)."""
    pp = Path(p).expanduser()
    # resolve() may fail on some systems for broken symlinks, so keep safe
    try:
        pp = pp.resolve()
    except Exception:
        pp = pp.absolute()
    return os.path.normcase(os.path.normpath(str(pp)))


def plane_to_one_hot(plane: str) -> torch.Tensor:
    """
    plane_one_hot: axial -> [0,1], coronal -> [1,0]
    """
    pl = (plane or "").lower().strip()
    if pl == "axial":
        return torch.tensor([0.0, 1.0], dtype=torch.float32)
    if pl == "coronal":
        return torch.tensor([1.0, 0.0], dtype=torch.float32)
    raise ValueError(f"Unknown plane='{plane}'. Expected 'axial' or 'coronal'.")


def infer_plane_from_path(path: str | Path, default_plane: str = "axial") -> str:
    """
    Light heuristic: look for tokens in path parts. Falls back to default_plane.
    Examples: .../axial/... or .../coronal/...
    """
    parts = [str(x).lower() for x in Path(path).parts]
    if any("coronal" in s for s in parts):
        return "coronal"
    if any("axial" in s for s in parts):
        return "axial"
    return default_plane


def load_label_map_from_csv(
    csv_path: str | Path,
    root_dir: Optional[str | Path] = None,
    path_col: str = "image_path",
    label_col: str = "label",
) -> Dict[str, int]:
    """
    CSV: maps image_path -> label (int).
    - Accepts absolute or relative paths.
    - If root_dir is provided, relative paths are interpreted relative to root_dir.
    - Keys are canonicalized for robust matching.
    """
    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"label_csv not found: {csv_path}")

    root_dir_path = Path(root_dir).expanduser() if root_dir else None

    label_map: Dict[str, int] = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if path_col not in reader.fieldnames or label_col not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain columns '{path_col}' and '{label_col}'. "
                f"Found columns: {reader.fieldnames}"
            )
        for row in reader:
            raw_p = (row.get(path_col) or "").strip()
            raw_y = (row.get(label_col) or "").strip()
            if not raw_p:
                continue
            if root_dir_path and not os.path.isabs(raw_p):
                p_full = root_dir_path / raw_p
            else:
                p_full = Path(raw_p)
            key = _canonicalize_path(p_full)
            # allow label as int-like string
            try:
                y = int(float(raw_y))
            except Exception:
                # if label is empty or non-numeric, skip
                continue
            label_map[key] = y
    return label_map


def _pil_to_tensor_gray01(img: "Image.Image", image_size: int) -> torch.Tensor:
    """
    Convert PIL image to torch tensor [1, H, W], float32 in [0,1], resized to image_size x image_size.
    """
    if img.mode != "L":
        img = img.convert("L")
    if image_size is not None:
        img = img.resize((int(image_size), int(image_size)), resample=Image.BILINEAR)
    # to tensor in [0,1]
    import numpy as np
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
    return t.clamp(0.0, 1.0)


# -------------------------
# Dataset
# -------------------------
@dataclass
class DatasetItem:
    input: torch.Tensor          # [1,H,W]
    path: str                    # original path string
    plane_one_hot: torch.Tensor  # [2]
    label: Optional[int] = None  # int if available


class FolderSubfolderImageDataset(Dataset):
    """
    Dataset that scans root_dir recursively for image files.

    Returns dict with keys:
      - "input": Float tensor [1,H,W] in [0,1]
      - "path":  str
      - "plane_one_hot": Float tensor [2]
      - "label": Long tensor scalar (optional, if label_csv provided and matched)

    Notes:
      - No preprocessing here by requirement.
      - Plane can be fixed (plane="axial"/"coronal") or inferred from path (plane="auto").
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 192,
        plane: str = "axial",              # "axial" | "coronal" | "auto"
        label_map: Optional[Dict[str, int]] = None,
        paths: Optional[Sequence[str | Path]] = None,
        image_loader: Optional[Callable[[Path, int], torch.Tensor]] = None,
    ):
        self.root_dir = Path(root_dir).expanduser()
        if not self.root_dir.exists():
            raise FileNotFoundError(f"data root not found: {self.root_dir}")

        self.image_size = int(image_size)
        self.plane_mode = plane
        self.label_map = label_map or {}
        self.image_loader = image_loader or (lambda p, sz: _pil_to_tensor_gray01(Image.open(p), sz))

        if paths is None:
            self.paths: List[Path] = sorted([p for p in self.root_dir.rglob("*") if _is_image_file(p)])
        else:
            self.paths = [Path(p).expanduser() for p in paths]

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {self.root_dir} with extensions: {sorted(_IMG_EXTS)}")

        # Precompute canonical keys to speed label lookup
        self._canon: List[str] = [_canonicalize_path(p) for p in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def _get_plane(self, path: Path) -> str:
        if (self.plane_mode or "").lower().strip() == "auto":
            return infer_plane_from_path(path, default_plane="axial")
        return self.plane_mode

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        p = self.paths[idx]
        x = self.image_loader(p, self.image_size)  # [1,H,W], float in [0,1]

        plane_str = self._get_plane(p)
        plane_oh = plane_to_one_hot(plane_str)

        key = self._canon[idx]
        y = self.label_map.get(key, None)

        item: Dict[str, torch.Tensor | str] = {
            "input": x,
            "path": str(p),
            "plane_one_hot": plane_oh,
        }
        if y is not None:
            item["label"] = torch.tensor(y, dtype=torch.long)
        return item


# -------------------------
# Split + DataLoaders
# -------------------------
def split_indices(
    n: int,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Random split indices into train/val/test.

    val_ratio and test_ratio are fractions in [0,1).
    """
    if n <= 0:
        return [], [], []
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("Require val_ratio>=0, test_ratio>=0, and val_ratio+test_ratio < 1.0")

    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)

    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_test = min(max(n_test, 0), n)
    n_val = min(max(n_val, 0), n - n_test)

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def select_indices_by_train_mod(n: int, train_mod: float = 1.0) -> List[int]:
    """
    Deterministic index subsampling for train_mod >= 1.

    - Integer train_mod=k: keep 0, k, 2k, ...
    - Float train_mod=m: keep approximately 1/m samples with interleaving spacing.
      Example m=2.5 -> 0,3,5,8,10,... (intervals 3,2,3,2,...)
    """
    if n <= 0:
        return []

    m = float(train_mod)
    if (not math.isfinite(m)) or (m < 1.0):
        raise ValueError("train_mod must be a finite number >= 1")

    if m == 1.0:
        return list(range(n))

    if m.is_integer():
        step = int(m)
        return list(range(0, n, step))

    selected: List[int] = []
    prev_bucket = -1
    for i in range(n):
        bucket = math.floor(i / m)
        if bucket > prev_bucket:
            selected.append(i)
            prev_bucket = bucket
    return selected


def create_dataloaders_from_folder(
    data_root: str | Path,
    train_mod: float = 1.0,
    image_size: int = 192,
    plane: str = "axial",               # axial|coronal|auto
    label_csv: Optional[str | Path] = None,
    label_path_col: str = "image_path",
    label_col: str = "label",
    batch_size: int = 64,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    drop_last: bool = True,
    split_test: bool = True,  # NEW
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], FolderSubfolderImageDataset]:
    """
    Build train/val/(optional)test dataloaders from a folder with subfolders.

    Returns: (train_loader, val_loader, test_loader_or_None, full_dataset)
    """
    label_map = None
    if label_csv:
        label_map = load_label_map_from_csv(
            csv_path=label_csv,
            root_dir=data_root,
            path_col=label_path_col,
            label_col=label_col,
        )

    if float(train_mod) < 1.0:
        raise ValueError("train_mod must be >= 1")

    root_dir = Path(data_root).expanduser()
    if not root_dir.exists():
        raise FileNotFoundError(f"data root not found: {root_dir}")
    all_paths = sorted([p for p in root_dir.rglob("*") if _is_image_file(p)])
    if len(all_paths) == 0:
        raise RuntimeError(f"No images found under {root_dir} with extensions: {sorted(_IMG_EXTS)}")

    selected_idx = select_indices_by_train_mod(len(all_paths), float(train_mod))
    selected_paths = [all_paths[i] for i in selected_idx]
    if len(selected_paths) == 0:
        raise RuntimeError(f"No images selected after applying train_mod={train_mod}")

    full_ds = FolderSubfolderImageDataset(
        root_dir=data_root,
        image_size=image_size,
        plane=plane,
        label_map=label_map,
        paths=selected_paths,
    )

    # NEW: nếu không split test thì ép test_ratio = 0 để không tạo test split
    eff_test_ratio = test_ratio if split_test else 0.0

    train_idx, val_idx, test_idx = split_indices(
        n=len(full_ds),
        val_ratio=val_ratio,
        test_ratio=eff_test_ratio,
        seed=seed,
    )

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    def _make_loader(ds, shuffle: bool) -> DataLoader:
        extra_kwargs = {}
        if num_workers > 0:
            extra_kwargs["persistent_workers"] = True
            extra_kwargs["prefetch_factor"] = 2
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last if shuffle else False,
            **extra_kwargs,
        )

    train_loader = _make_loader(train_ds, shuffle=True)
    val_loader = _make_loader(val_ds, shuffle=False)

    if split_test:
        test_ds = Subset(full_ds, test_idx) if len(test_idx) > 0 else Subset(full_ds, val_idx)
        test_loader: Optional[DataLoader] = _make_loader(test_ds, shuffle=False)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader, full_ds


__all__ = [
    "FolderSubfolderImageDataset",
    "create_dataloaders_from_folder",
    "select_indices_by_train_mod",
    "load_label_map_from_csv",
    "plane_to_one_hot",
    "infer_plane_from_path",
]
