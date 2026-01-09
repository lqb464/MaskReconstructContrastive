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
