from __future__ import annotations

import argparse

"""
Skull stripping experiment shim.
Recon-only is the default path in this package; masking/contrastive flags are retained for CLI compatibility.
"""

from ..config.experiment import ExperimentConfig as _BaseExperimentConfig
from ..config.experiment import build_argparser as _build_argparser

UNUSED_IN_RECON_ONLY = (
    "enable_masking",
    "mask_ratio",
    "lambda_contrast",
    "contrastive_loss_type",
    "contrastive_position",
)

class ExperimentConfig(_BaseExperimentConfig):
    pass

def build_argparser():
    parser = _build_argparser()

    parser.set_defaults(enable_contrastive=False, enable_masking=False)
    return parser

def enforce_recon_only_args(args: argparse.Namespace) -> None:
    if bool(getattr(args, "enable_contrastive", False)):
        raise ValueError(
            "skull_stripping/main.py is reconstruction-only. "
            "Disable contrastive with --disable-contrastive (or remove --enable-contrastive)."
        )

__all__ = ["ExperimentConfig", "build_argparser", "enforce_recon_only_args", "UNUSED_IN_RECON_ONLY"]
