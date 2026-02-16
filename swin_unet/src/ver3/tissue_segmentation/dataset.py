from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ..data.dataset import plane_to_one_hot
from ..mask_reconstruction.pair_transforms import apply_pair_transforms, load_image_pil
from .io import LabelEncodingInfo, encode_label_array, load_label_array, resolve_scan_tokens_to_images

log = logging.getLogger(__name__)


class TissueSegmentationDataset(Dataset):
    """
    Tissue segmentation dataset with separate image and label roots.

    Contract:
      - input: float32 [1,H,W] in [0,1]
      - target: int64 [H,W] encoded class ids
      - path: str image path
      - plane_one_hot: float32 [2]
    """

    def __init__(
        self,
        *,
        image_root: str | Path,
        label_root: str | Path,
        scan_tokens: List[str],
        encoding_info: LabelEncodingInfo,
        image_ext: str = ".png",
        label_suffix: str = "_label.npz",
        label_key: Optional[str] = None,
        image_size: Optional[int] = None,
        target_size: int = 0,
        resize_mode: str = "letterbox",
        plane: str = "axial",
        strict_pairs: bool = False,
        debug_shapes: bool = False,
    ):
        self.image_root = Path(image_root).expanduser().resolve()
        self.label_root = Path(label_root).expanduser().resolve()
        if not self.image_root.exists():
            raise FileNotFoundError(f"image_root not found: {self.image_root}")
        if not self.label_root.exists():
            raise FileNotFoundError(f"label_root not found: {self.label_root}")

        self.encoding_info = encoding_info
        self.image_ext = image_ext.lower()
        self.label_suffix = label_suffix
        self.label_key = label_key
        self.image_size = image_size
        self.target_size = int(target_size)
        self.resize_mode = resize_mode
        self.strict_pairs = bool(strict_pairs)
        self.debug_shapes = bool(debug_shapes)
        self.plane_one_hot = plane_to_one_hot(plane).contiguous()

        self.images = resolve_scan_tokens_to_images(
            image_root=self.image_root,
            tokens=scan_tokens,
            image_ext=self.image_ext,
        )
        if not self.images:
            raise RuntimeError("No image paths resolved from provided scan list.")

        # Fallback index by basename stem for labels: <stem>_label.npz
        self._label_stem_index = self._build_label_stem_index()

        self.pairs: List[Tuple[Path, Path]] = []
        missing_labels: List[Path] = []
        for img_path in self.images:
            lbl_path = self._resolve_label_path(img_path)
            if lbl_path is None:
                missing_labels.append(img_path)
                continue
            self.pairs.append((img_path, lbl_path))

        if missing_labels:
            sample = ", ".join(x.name for x in missing_labels[:5])
            msg = (
                f"{len(missing_labels)} images have no matching label and were filtered out "
                f"(example: {sample})."
            )
            if self.strict_pairs:
                raise FileNotFoundError(msg)
            log.warning(msg)

        if not self.pairs:
            raise RuntimeError(
                "No image/label pairs remain after filtering for existing labels. "
                "Check image_root, label_root, list file, and label_suffix."
            )

    def _build_label_stem_index(self) -> Dict[str, List[Path]]:
        """
        Pre-index labels by stem extracted from <stem><label_suffix>.
        """
        suffix = self.label_suffix
        index: Dict[str, List[Path]] = {}
        for p in sorted(self.label_root.rglob("*")):
            if not p.is_file():
                continue
            if not p.name.endswith(suffix):
                continue
            stem = p.name[: -len(suffix)]
            index.setdefault(stem.lower(), []).append(p.resolve())
        return index

    def _resolve_label_path(self, image_path: Path) -> Optional[Path]:
        # Strategy 1: mirrored relative tree under label_root.
        rel = image_path.resolve().relative_to(self.image_root)
        c1 = self.label_root / rel.parent / f"{image_path.stem}{self.label_suffix}"
        if c1.exists():
            return c1.resolve()

        # Strategy 2: deterministic basename stem fallback.
        cands = self._label_stem_index.get(image_path.stem.lower(), [])
        if not cands:
            return None
        return sorted(cands)[0]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        img_path, lbl_path = self.pairs[idx]

        img_pil = load_image_pil(img_path)
        lbl_np = load_label_array(lbl_path, key=self.label_key)

        if self.target_size > 0:
            target_sz = int(self.target_size)
        elif self.image_size is not None and int(self.image_size) > 0:
            target_sz = int(self.image_size)
        else:
            w, h = img_pil.size
            target_sz = int(max(w, h))

        x, y_ids = apply_pair_transforms(
            img_pil,
            lbl_np,
            target_sz,
            do_hflip=False,
            resize_mode=self.resize_mode,
        )

        y_enc = encode_label_array(y_ids.squeeze(0).cpu().numpy(), self.encoding_info)
        y = torch.from_numpy(y_enc).to(dtype=torch.long)

        if self.debug_shapes and idx < 3:
            print(
                f"[tissue_dataset] idx={idx} input_hw={tuple(x.shape[-2:])} "
                f"target_hw={tuple(y.shape[-2:])} num_classes={self.encoding_info.num_classes}"
            )

        return {
            "input": x,
            "target": y,
            "path": str(img_path),
            "plane_one_hot": self.plane_one_hot,
        }


__all__ = ["TissueSegmentationDataset"]
