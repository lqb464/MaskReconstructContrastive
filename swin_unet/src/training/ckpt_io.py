from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch

def save_checkpoint(
    *,
    path: Path,
    epoch: int,
    best_val: float,
    model,
    optimizer,
    scaler,
    cfg,
) -> None:
    """Save checkpoint with the same format as legacy trainer."""
    obj = {
        "epoch": epoch,
        "best_val": float(best_val),
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg": asdict(cfg),
    }
    torch.save(obj, path)

def load_checkpoint_weights(
    *,
    ckpt_path: Path,
    device: torch.device,
    model,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load only model weights from checkpoint; return full checkpoint object."""
    obj = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(obj["model"], strict=strict)
    return obj

def load_checkpoint_weights_filtered(
    *,
    ckpt_path: Path,
    device: torch.device,
    model,
    include_prefixes: tuple[str, ...] | None = None,
    exclude_prefixes: tuple[str, ...] | None = None,
) -> Dict[str, Any]:
    """
    Load model weights from checkpoint with key filtering.
    Always loads with strict=False.
    Adds obj["_load_msg"] containing missing_keys and unexpected_keys.
    """
    obj = torch.load(ckpt_path, map_location=device)
    sd = obj["model"]

    if include_prefixes is not None:
        sd = {k: v for k, v in sd.items() if k.startswith(include_prefixes)}

    if exclude_prefixes is not None:
        sd = {k: v for k, v in sd.items() if not k.startswith(exclude_prefixes)}

    msg = model.load_state_dict(sd, strict=False)
    obj["_load_msg"] = {"missing_keys": msg.missing_keys, "unexpected_keys": msg.unexpected_keys}
    return obj
