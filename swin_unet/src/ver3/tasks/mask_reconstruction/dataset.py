from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict

import torch
from torch.utils.data import Dataset
import numpy as np

from ...data.dataset import plane_to_one_hot
from .pair_transforms import (
    load_image_pil,
    apply_pair_transforms,
)
from .io import load_mask_npz_array

log = logging.getLogger(__name__)

_DEBUG_PAIR_ALIGNMENT_ENV = bool(int(os.getenv("MASK_RECON_DEBUG_PAIR_ALIGNMENT", "0")))
_SOBEL_X_KERNEL = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], dtype=torch.float32)
_SOBEL_Y_KERNEL = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]], dtype=torch.float32)
_MASK_EDGE_KERNEL = torch.ones((1, 1, 3, 3), dtype=torch.float32)


class MaskReconstructionDataset(Dataset):
    """
    Mask reconstruction dataset.
    Unlike the generic folder/subfolder classification dataset, this class enforces image-mask pairing
    and returns reconstruction targets for each input slice.

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
        return_dual_view: bool = False,
        debug_pair_alignment: bool = False,
        plane: str = "axial",
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
        self.return_dual_view = return_dual_view
        self.debug_pair_alignment = bool(debug_pair_alignment) and _DEBUG_PAIR_ALIGNMENT_ENV
        self.debug_pair_alignment_mod = max(1, int(os.getenv("MASK_RECON_DEBUG_PAIR_ALIGNMENT_MOD", "64")))

        self.plane_one_hot = plane_to_one_hot(plane)
        if self.debug_shapes:
            print(f"[dataset] plane={plane} one_hot={self.plane_one_hot.tolist()}")
        if bool(debug_pair_alignment) and not _DEBUG_PAIR_ALIGNMENT_ENV:
            log.info("debug_pair_alignment requested but MASK_RECON_DEBUG_PAIR_ALIGNMENT=0, so debug is disabled.")

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
        img_pil = load_image_pil(img_path)
        mask_np = load_mask_npz_array(mask_path, key=self.mask_key)

        if self.target_size > 0:
            target_sz = int(self.target_size)
        elif self.image_size is not None and self.image_size > 0:
            target_sz = int(self.image_size)
        else:
            w, h = img_pil.size
            target_sz = int(max(w, h))

        x, y = apply_pair_transforms(img_pil, mask_np, target_sz, do_hflip=False, resize_mode=self.resize_mode)
        y = (y > 0).float()

        # Debug alignment logging is opt-in and sampled to avoid throughput collapse.
        if self.debug_pair_alignment and (idx % self.debug_pair_alignment_mod == 0):
            bbox = (y[0] > 0).nonzero(as_tuple=False)
            if bbox.numel() > 0:
                rmin, cmin = bbox[:, 0].min().item(), bbox[:, 1].min().item()
                rmax, cmax = bbox[:, 0].max().item(), bbox[:, 1].max().item()
            else:
                rmin = cmin = rmax = cmax = -1
            pad_frac = float((x < 1e-3).float().mean().item())
            mask_hw = tuple(mask_np.shape[-2:]) if mask_np.ndim >= 2 else tuple(mask_np.shape)
            print(
                f"[pair_debug] idx={idx} img={img_path.name} orig_hw={img_pil.size[::-1]} "
                f"mask_hw={mask_hw} target_sz={target_sz} resize_mode={self.resize_mode} hflip=False "
                f"tensor_hw={tuple(x.shape[-2:])} mask_bbox={(rmin, rmax, cmin, cmax)} pad_frac~{pad_frac:.3f}"
            )
            # crude edge vs mask boundary overlap (IoU proxy)
            sobel_x = torch.nn.functional.conv2d(
                x.unsqueeze(0),
                weight=_SOBEL_X_KERNEL.to(device=x.device, dtype=x.dtype),
                padding=1,
            )
            sobel_y = torch.nn.functional.conv2d(
                x.unsqueeze(0),
                weight=_SOBEL_Y_KERNEL.to(device=x.device, dtype=x.dtype),
                padding=1,
            )
            edges = (sobel_x.abs() + sobel_y.abs()).squeeze(0)
            edge_mask = (edges > edges.mean()).float()
            mask_bound = torch.nn.functional.conv2d(
                y.unsqueeze(0),
                weight=_MASK_EDGE_KERNEL.to(device=y.device, dtype=y.dtype),
                padding=1,
            ).squeeze(0)
            mask_edge = ((mask_bound > 0) & (mask_bound < 9)).float()
            inter = (edge_mask * mask_edge).sum()
            union = (edge_mask + mask_edge - edge_mask * mask_edge).sum().clamp(min=1.0)
            iou_proxy = (inter / union).item()
            if iou_proxy < 0.01:
                print(f"[pair_debug] warning: low edge/mask overlap (IoU~{iou_proxy:.4f}) for {img_path.name}")

        if self.return_dual_view:
            # Reuse already transformed tensors; mirror in tensor space to avoid duplicate resize/convert work.
            x2 = torch.flip(x, dims=[-1])
            y2 = torch.flip(y, dims=[-1])
            assert x2.shape[-2:] == y2.shape[-2:], f"Shape mismatch after transforms view2: {x2.shape} vs {y2.shape}"
        else:
            x2, y2 = None, None

        if self.debug_shapes and idx < 3:
            print(f"[debug] sample {idx}: shape {x.shape[-2:]} target_sz={target_sz} mode={self.resize_mode}")

        assert x.shape[-2:] == y.shape[-2:], f"Shape mismatch after transforms: {x.shape} vs {y.shape}"
        assert x.ndim == 3 and y.ndim == 3 and x.shape[0] == 1 and y.shape[0] == 1, f"Unexpected tensor shapes: {x.shape}, {y.shape}"

        if self.return_dual_view:
            return {
                "input1": x,
                "target1": y,
                "input2": x2,
                "target2": y2,
                "path": str(img_path),
                "plane_one_hot": self.plane_one_hot.clone(),
            }

        return {
            "input": x,
            "target": y,
            "path": str(img_path),
            "plane_one_hot": self.plane_one_hot.clone(),
        }

__all__ = ["MaskReconstructionDataset"]
