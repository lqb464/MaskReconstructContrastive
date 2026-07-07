from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_CFG_WARN_FIELDS = (
    ("model", "backbone"),
    ("model", "base_ch"),
    ("data", "image_size"),
    ("training", "batch_size"),
    ("training", "lr"),
)


def warn_cfg_mismatch(*, saved_cfg: Optional[Dict[str, Any]], current_cfg) -> None:
    """Warn when key hyperparameters differ between checkpoint cfg and current run."""
    if not saved_cfg:
        return
    for section, key in _CFG_WARN_FIELDS:
        saved_section = saved_cfg.get(section, {})
        current_section = asdict(current_cfg).get(section, {})
        saved_val = saved_section.get(key)
        current_val = current_section.get(key)
        if saved_val is not None and current_val is not None and saved_val != current_val:
            print(f"[ckpt] warning: cfg mismatch {section}.{key}: ckpt={saved_val!r} current={current_val!r}")

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


def load_checkpoint_training_state(
    *,
    ckpt_path: Path,
    device: torch.device,
    model,
    optimizer,
    scaler,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load full training state for resume: model, optimizer, scaler, epoch, best_val.
    Returns dict with start_epoch (epoch + 1) and best_val.
    """
    obj = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(obj["model"], strict=strict)

    start_epoch = 1
    best_val = float("inf")
    if "epoch" in obj:
        start_epoch = int(obj["epoch"]) + 1
    if "best_val" in obj:
        best_val = float(obj["best_val"])

    if optimizer is not None and "opt" in obj:
        try:
            optimizer.load_state_dict(obj["opt"])
        except Exception as e:
            print(f"[ckpt] warning: could not load optimizer state ({e!r}); using fresh optimizer")

    if scaler is not None and getattr(scaler, "is_enabled", lambda: True)() and "scaler" in obj:
        try:
            scaler.load_state_dict(obj["scaler"])
        except Exception as e:
            print(f"[ckpt] warning: could not load scaler state ({e!r}); using fresh scaler")

    return {
        "start_epoch": start_epoch,
        "best_val": best_val,
        "cfg": obj.get("cfg"),
        "saved_epoch": int(obj.get("epoch", 0)),
    }
