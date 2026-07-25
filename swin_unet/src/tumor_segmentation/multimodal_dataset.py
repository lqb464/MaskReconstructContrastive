from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from ..data.dataset import infer_plane_from_path, plane_to_one_hot
from ..skull_stripping.pair_transforms import apply_pair_transforms, load_image_pil
from ..tissue_segmentation.io import (
    ImageIndex,
    LabelEncodingInfo,
    encode_label_array,
    load_label_array,
)
from .scan_lists import ModalityStackSample, build_modality_stack_samples

log = logging.getLogger(__name__)


class ModalityStackSegmentationDataset(Dataset):
    """
    Tumor segmentation dataset that stacks multiple MRI modalities as input channels.

    Each sample is one (patient, slice) with shape [C, H, W] where C = len(modalities).
    Label is shared across modalities (same mask stem per prepare_brats2021.py).
    """

    def __init__(
        self,
        *,
        image_root: str | Path,
        label_root: str | Path,
        patient_tokens: Sequence[str],
        stack_modalities: Sequence[str],
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

        self.stack_modalities = [str(m).lower() for m in stack_modalities]
        if len(self.stack_modalities) < 2:
            raise ValueError(
                f"stack_modalities must contain at least 2 entries, got {self.stack_modalities}"
            )

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

        if image_index is None:
            raise ValueError("ModalityStackSegmentationDataset requires a pre-built image_index.")
        self.num_images_resolved = int(len(image_index.all_images))

        stack_samples, skipped_incomplete = build_modality_stack_samples(
            image_paths=image_index.all_images,
            patient_tokens=patient_tokens,
            modalities=self.stack_modalities,
        )
        self.num_incomplete_stack_groups = int(skipped_incomplete)

        self._label_stem_index = self._build_label_stem_index()
        self.pairs: List[Tuple[ModalityStackSample, Path]] = []
        missing_labels: List[str] = []
        for sample in stack_samples:
            lbl_path = self._resolve_label_path(sample.reference_image_path)
            if lbl_path is None:
                missing_labels.append(sample.group_key)
                continue
            self.pairs.append((sample, lbl_path))

        self.num_missing_labels = int(len(missing_labels))
        self.num_labeled_samples = int(len(self.pairs))

        if missing_labels:
            sample = ", ".join(missing_labels[:5])
            msg = (
                f"{len(missing_labels)} stacked slice groups have no matching label and were filtered out "
                f"(example: {sample})."
            )
            if self.strict_pairs:
                raise FileNotFoundError(msg)
            log.warning(msg)

        if not self.pairs:
            raise RuntimeError(
                "No stacked modality groups remain after filtering for labels. "
                "Check --modality, patient lists, and prepare naming "
                "(*_{mod}_z####.png with matching labels)."
            )

    def _build_label_stem_index(self) -> Dict[str, List[Path]]:
        suffix = self.label_suffix
        index: Dict[str, List[Path]] = {}
        for p in sorted(self.label_root.rglob("*")):
            if not p.is_file() or not p.name.endswith(suffix):
                continue
            stem = p.name[: -len(suffix)]
            index.setdefault(stem.lower(), []).append(p.resolve())
        return index

    def _resolve_label_path(self, image_path: Path) -> Optional[Path]:
        rel = image_path.resolve().relative_to(self.image_root)
        c1 = self.label_root / rel.parent / f"{image_path.stem}{self.label_suffix}"
        if c1.exists():
            return c1.resolve()

        # Try unified label path (strip modality suffix: BraTS2021_00000_flair_z0027 -> BraTS2021_00000_z0027)
        import re
        match = re.match(r"^(.+)_(t1ce|flair|t1|t2|cer|hyper)_(z\d+)$", image_path.stem, re.IGNORECASE)
        if match:
            patient, _, slice_tag = match.groups()
            unified_stem = f"{patient}_{slice_tag}"
            c2 = self.label_root / rel.parent / f"{unified_stem}{self.label_suffix}"
            if c2.exists():
                return c2.resolve()
            cands_u = self._label_stem_index.get(unified_stem.lower(), [])
            if cands_u:
                return sorted(cands_u)[0]

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
        sample, lbl_path = self.pairs[idx]
        lbl_np = load_label_array(lbl_path, key=self.label_key)

        if self.target_size > 0:
            target_sz = int(self.target_size)
        elif self.image_size is not None and int(self.image_size) > 0:
            target_sz = int(self.image_size)
        else:
            ref_pil = load_image_pil(sample.reference_image_path)
            w, h = ref_pil.size
            target_sz = int(max(w, h))

        channel_tensors: List[torch.Tensor] = []
        y: torch.Tensor | None = None
        for mod_idx, (_mod, img_path) in enumerate(sample.modality_paths):
            img_pil = load_image_pil(img_path)
            img_t, lbl_t = apply_pair_transforms(
                img_pil,
                lbl_np,
                target_sz,
                do_hflip=False,
                resize_mode=self.resize_mode,
            )
            channel_tensors.append(img_t)
            if mod_idx == 0:
                y_enc = encode_label_array(
                    lbl_t.squeeze(0).cpu().numpy(),
                    self.encoding_info,
                    strict_label_ids=self.strict_label_ids,
                    allow_unknown_label_ids=self.allow_unknown_label_ids,
                    unknown_fallback_id=0,
                )
                y = torch.from_numpy(y_enc).to(dtype=torch.long)

        assert y is not None
        x = torch.cat(channel_tensors, dim=0)

        expected_ch = len(self.stack_modalities)
        if x.dtype != torch.float32:
            raise TypeError(f"Dataset contract violation: input dtype must be float32, got {x.dtype}")
        if x.ndim != 3 or x.shape[0] != expected_ch:
            raise ValueError(
                f"Dataset contract violation: input must have shape [{expected_ch},H,W], got {tuple(x.shape)}"
            )

        if y.dtype != torch.long or y.ndim != 2:
            raise ValueError(f"Dataset contract violation: target must be torch.long [H,W], got {y.dtype} {tuple(y.shape)}")

        if self.debug_shapes and idx < 3:
            print(
                f"[multimodal_dataset] idx={idx} group={sample.group_key} "
                f"modalities={self.stack_modalities} input_hw={tuple(x.shape[-2:])}"
            )

        return {
            "input": x,
            "target": y,
            "path": str(sample.reference_image_path),
            "plane_one_hot": self._plane_one_hot_for_path(sample.reference_image_path),
        }

    def dataset_summary(self) -> Dict[str, object]:
        return {
            "num_samples": int(len(self.pairs)),
            "num_images_resolved": int(self.num_images_resolved),
            "num_labeled_samples": int(self.num_labeled_samples),
            "num_missing_labels": int(self.num_missing_labels),
            "num_incomplete_stack_groups": int(self.num_incomplete_stack_groups),
            "stack_modalities": list(self.stack_modalities),
            "input_channels": int(len(self.stack_modalities)),
            "num_classes": int(self.encoding_info.num_classes),
            "plane_mode": self.plane_mode,
        }


__all__ = ["ModalityStackSegmentationDataset"]
