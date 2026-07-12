from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ..data.dataset import infer_plane_from_path, plane_to_one_hot
from ..skull_stripping.pair_transforms import apply_pair_transforms, load_image_pil
from .io import (
    ImageIndex,
    LabelEncodingInfo,
    encode_label_array,
    load_label_array,
    resolve_scan_tokens_to_images,
)

log = logging.getLogger(__name__)

class TissueSegmentationDataset(Dataset):
    """
    Tissue segmentation dataset with separate image and label roots.

    __getitem__ contract:
      - "input":
        dtype: torch.float32
        shape: [1, H, W]
        range: [0, 1] (small numerical tolerance allowed)
      - "target":
        dtype: torch.int64 (torch.long)
        shape: [H, W]
        range: [0, num_classes-1]
      - "path":
        dtype: str
      - "plane_one_hot":
        dtype: torch.float32
        shape: [2]
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
        strict_label_ids: bool = True,
        allow_unknown_label_ids: bool = False,
        debug_shapes: bool = False,
        image_index: ImageIndex | None = None,
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
        self.strict_label_ids = bool(strict_label_ids)
        self.allow_unknown_label_ids = bool(allow_unknown_label_ids)
        self.debug_shapes = bool(debug_shapes)
        self._plane_one_hot_lut = {
            "axial": plane_to_one_hot("axial").contiguous(),
            "coronal": plane_to_one_hot("coronal").contiguous(),
        }
        self.plane_mode = str(plane).lower().strip()
        if self.plane_mode not in {"axial", "coronal", "auto"}:
            raise ValueError(f"Unknown plane='{plane}'. Expected one of: axial, coronal, auto")
        self._fixed_plane_one_hot = self._plane_one_hot_lut.get(self.plane_mode, None)

        self.images = resolve_scan_tokens_to_images(
            image_root=self.image_root,
            tokens=scan_tokens,
            image_ext=self.image_ext,
            image_index=image_index,
        )
        if not self.images:
            sample_tokens = ", ".join(scan_tokens[:5]) if scan_tokens else "<empty>"
            raise RuntimeError(
                "No image paths resolved from provided scan list. "
                f"image_root={self.image_root} image_ext={self.image_ext} sample_tokens=[{sample_tokens}]"
            )
        self.num_images_resolved = int(len(self.images))

        self._label_stem_index = self._build_label_stem_index()

        self.pairs: List[Tuple[Path, Path]] = []
        missing_labels: List[Path] = []
        for img_path in self.images:
            lbl_path = self._resolve_label_path(img_path)
            if lbl_path is None:
                missing_labels.append(img_path)
                continue
            self.pairs.append((img_path, lbl_path))
        self.num_missing_labels = int(len(missing_labels))
        self.num_labeled_samples = int(len(self.pairs))

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

        rel = image_path.resolve().relative_to(self.image_root)
        c1 = self.label_root / rel.parent / f"{image_path.stem}{self.label_suffix}"
        if c1.exists():
            return c1.resolve()

        cands = self._label_stem_index.get(image_path.stem.lower(), [])
        if not cands:
            return None
        return sorted(cands)[0]

    def __len__(self) -> int:
        return len(self.pairs)

    def _plane_one_hot_for_path(self, image_path: Path) -> torch.Tensor:
        if self._fixed_plane_one_hot is not None:
            return self._fixed_plane_one_hot
        inferred = infer_plane_from_path(image_path, default_plane="axial")
        return self._plane_one_hot_lut.get(inferred, self._plane_one_hot_lut["axial"])

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

        y_enc = encode_label_array(
            y_ids.squeeze(0).cpu().numpy(),
            self.encoding_info,
            strict_label_ids=self.strict_label_ids,
            allow_unknown_label_ids=self.allow_unknown_label_ids,
            unknown_fallback_id=0,
        )
        y = torch.from_numpy(y_enc).to(dtype=torch.long)

        if x.dtype != torch.float32:
            raise TypeError(f"Dataset contract violation: input dtype must be float32, got {x.dtype}")
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"Dataset contract violation: input must have shape [1,H,W], got {tuple(x.shape)}")
        x_min = float(x.min().item()) if x.numel() > 0 else 0.0
        x_max = float(x.max().item()) if x.numel() > 0 else 0.0
        tol = 1e-4
        if x_min < -tol or x_max > 1.0 + tol:
            raise ValueError(f"Dataset contract violation: input range out of [0,1] with tol={tol}: min={x_min}, max={x_max}")

        if y.dtype != torch.long:
            raise TypeError(f"Dataset contract violation: target dtype must be torch.long, got {y.dtype}")
        if y.ndim != 2:
            raise ValueError(f"Dataset contract violation: target must have shape [H,W], got {tuple(y.shape)}")
        y_min = int(y.min().item()) if y.numel() > 0 else 0
        y_max = int(y.max().item()) if y.numel() > 0 else 0
        if y_min < 0 or y_max >= int(self.encoding_info.num_classes):
            raise ValueError(
                f"Dataset contract violation: target ids out of [0,{self.encoding_info.num_classes - 1}] "
                f"(min={y_min}, max={y_max})"
            )

        if self.debug_shapes and idx < 3:
            print(
                f"[tissue_dataset] idx={idx} input_hw={tuple(x.shape[-2:])} "
                f"target_hw={tuple(y.shape[-2:])} num_classes={self.encoding_info.num_classes}"
            )

        return {
            "input": x,
            "target": y,
            "path": str(img_path),
            "plane_one_hot": self._plane_one_hot_for_path(img_path),
        }

    def dataset_summary(self) -> Dict[str, object]:
        """
        Deterministic dataset-level metadata for audits/reporting.
        """
        merged_to_bg: list[int] = []
        for src_id, enc_id in self.encoding_info.encode_map.items():
            if int(enc_id) == 0 and int(src_id) != 0:
                merged_to_bg.append(int(src_id))

        return {
            "num_samples": int(len(self.pairs)),
            "num_images_resolved": int(self.num_images_resolved),
            "num_labeled_samples": int(self.num_labeled_samples),
            "num_missing_labels": int(self.num_missing_labels),
            "num_classes": int(self.encoding_info.num_classes),
            "orig_to_enc": dict(self.encoding_info.encode_map),
            "enc_to_orig": dict(self.encoding_info.decode_map),
            "unknown_ids": sorted(self.encoding_info.unknown_ids),
            "non_brain_ids": sorted(self.encoding_info.non_brain_ids),
            "merged_to_background": sorted(merged_to_bg),
            "excluded_ids_default_policy": [0],
            "strict_label_ids": bool(self.strict_label_ids),
            "allow_unknown_label_ids": bool(self.allow_unknown_label_ids),
            "plane_mode": self.plane_mode,
        }

__all__ = ["TissueSegmentationDataset"]
