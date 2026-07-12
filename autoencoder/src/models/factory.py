from __future__ import annotations

import torch.nn as nn

from ..config.experiment import ExperimentConfig
from .mae_dualview_ssl import MAEDualViewSSL
from .vae_dualview_ssl import VAEDualViewSSL


def build_model(cfg: ExperimentConfig, *, out_ch: int = 1) -> nn.Module:
    mcfg = cfg.model
    tcfg = cfg.training
    backbone = str(getattr(mcfg, "backbone", "mae")).lower()

    common = dict(
        in_ch=mcfg.in_ch,
        enable_reconstruct=tcfg.enable_reconstruct,
        single_view=tcfg.single_view,
    )

    if backbone == "mae":
        return MAEDualViewSSL(
            image_size=cfg.data.image_size,
            patch_size=mcfg.patch_size,
            embed_dim=mcfg.embed_dim,
            enc_depth=int(mcfg.mae_enc_depth),
            dec_depth=int(mcfg.mae_dec_depth),
            out_ch=int(out_ch),
            base_ch=int(mcfg.base_ch),
            use_gn=bool(mcfg.use_gn),
            **common,
        )

    if backbone == "vae":
        return VAEDualViewSSL(
            base_ch=int(mcfg.base_ch),
            latent_dim=int(mcfg.latent_dim),
            use_gn=bool(mcfg.use_gn),
            out_ch=int(out_ch),
            **common,
        )

    raise ValueError(f"Unknown backbone: {backbone!r}. Expected one of: mae, vae.")


__all__ = ["build_model"]
