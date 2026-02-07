from __future__ import annotations

"""
Mask reconstruction experiment shim.
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
    return _build_argparser()


__all__ = ["ExperimentConfig", "build_argparser", "UNUSED_IN_RECON_ONLY"]

