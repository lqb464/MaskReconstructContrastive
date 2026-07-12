from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from ..config.experiment import ExperimentConfig
from ..training.ckpt_io import load_checkpoint_weights, load_checkpoint_weights_filtered


def load_pretrained_for_downstream(model: nn.Module, cfg: ExperimentConfig, *, device) -> None:
    resume_ckpt = str(getattr(cfg.training, "resume_ckpt", "") or "").strip()
    ckpt_mode = str(getattr(cfg.training, "ckpt_load_mode", "none") or "none").lower()
    if not resume_ckpt or ckpt_mode == "none":
        return

    ckpt_path = Path(resume_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"resume_ckpt not found: {ckpt_path}")

    if ckpt_mode == "full":
        obj = load_checkpoint_weights(
            ckpt_path=ckpt_path,
            device=device,
            model=model,
            strict=False,
        )
        print(f"[ckpt] loaded full weights from {ckpt_path} (epoch={obj.get('epoch', '?')})")
        return

    if ckpt_mode == "encoder_only":
        obj = load_checkpoint_weights_filtered(
            ckpt_path=ckpt_path,
            device=device,
            model=model,
            include_prefixes=model.encoder_state_dict_prefixes(),
        )
        msg = obj.get("_load_msg", {})
        print(f"[ckpt] loaded encoder_only from {ckpt_path}")
        print(f"[ckpt] missing_keys: {len(msg.get('missing_keys', []))}")
        print(f"[ckpt] unexpected_keys: {len(msg.get('unexpected_keys', []))}")
        return

    raise ValueError(f"Unsupported ckpt_load_mode: {ckpt_mode!r}")


__all__ = ["load_pretrained_for_downstream"]
