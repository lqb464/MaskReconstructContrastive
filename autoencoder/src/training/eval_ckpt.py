from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, get_args, get_origin, get_type_hints

import torch


def dataclass_from_dict(dc_type, raw: dict):
    """
    Rebuild nested dataclass from dict, resolving forward refs via get_type_hints().
    """
    if not is_dataclass(dc_type):
        raise TypeError(f"{dc_type} is not a dataclass")

    type_hints = get_type_hints(dc_type)

    kwargs = {}
    for f in fields(dc_type):
        name = f.name
        if name not in raw:
            continue

        val = raw[name]
        ftype = type_hints.get(name, f.type)

        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[name] = dataclass_from_dict(ftype, val)
            continue

        origin = get_origin(ftype)
        args = get_args(ftype)
        if origin is Union and isinstance(val, dict):
            dc_candidates = [a for a in args if is_dataclass(a)]
            if dc_candidates:
                kwargs[name] = dataclass_from_dict(dc_candidates[0], val)
                continue

        kwargs[name] = val

    return dc_type(**kwargs)


def resolve_ckpt_path(ckpt: str, ckpt_dir: Optional[str]) -> Path:
    """
    ckpt: "best" | "latest" | path
    """
    s = (ckpt or "").strip()
    if s.lower() in {"best", "latest"}:
        if not ckpt_dir:
            raise ValueError("ckpt_dir is empty in checkpoint cfg, cannot resolve 'best' or 'latest'")
        p = Path(ckpt_dir) / "checkpoints" / f"{s.lower()}.pt"

        if p.exists():
            return p
        p2 = Path(ckpt_dir) / f"{s.lower()}.pt"
        if p2.exists():
            return p2
        raise FileNotFoundError(f"Cannot resolve checkpoint '{s}' under: {ckpt_dir}")
    return Path(s).expanduser()


def load_checkpoint(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    obj = torch.load(ckpt_path, map_location=device)
    if not isinstance(obj, dict) or "model" not in obj:
        raise ValueError(f"Invalid checkpoint format: {ckpt_path}")
    return obj


__all__ = [
    "dataclass_from_dict",
    "resolve_ckpt_path",
    "load_checkpoint",
]
