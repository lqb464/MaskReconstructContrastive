from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import torch

from ..training.ckpt_io import save_checkpoint as _save_checkpoint

log = logging.getLogger(__name__)

def save_checkpoint(*args, **kwargs):
    return _save_checkpoint(*args, **kwargs)

def load_checkpoint_weights_filtered(
    model: torch.nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    state_key: str = "model",
    prefix_to_strip: str = "",
    require_no_missing: bool = False,
) -> Any:
    """
    Load model weights with strict=False and emit loud diagnostics for partial loads.
    Use require_no_missing=True for fail-fast behavior in recon-only training.
    """
    ckpt = torch.load(Path(path), map_location=map_location)
    if isinstance(ckpt, dict) and state_key in ckpt and isinstance(ckpt[state_key], dict):
        state_dict = ckpt[state_key]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint structure from {path}")

    if prefix_to_strip:
        plen = len(prefix_to_strip)
        state_dict = {k[plen:] if k.startswith(prefix_to_strip) else k: v for k, v in state_dict.items()}

    load_msg = model.load_state_dict(state_dict, strict=False)
    missing = list(load_msg.missing_keys)
    unexpected = list(load_msg.unexpected_keys)
    if missing or unexpected:
        log.warning("Checkpoint load report (%s): missing_keys=%d unexpected_keys=%d", path, len(missing), len(unexpected))
        if missing:
            msg = (
                f"Checkpoint missing keys ({len(missing)}): {missing}. "
                "For strict recon-only loads, pass require_no_missing=True. "
                "For intentional partial loads, keep require_no_missing=False."
            )
            if require_no_missing:
                raise RuntimeError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        if unexpected:
            warnings.warn(
                f"Checkpoint has unexpected keys ({len(unexpected)}): {unexpected}.",
                RuntimeWarning,
                stacklevel=2,
            )
    return load_msg

__all__ = ["save_checkpoint", "load_checkpoint_weights_filtered"]
