from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set python, numpy, torch RNG seeds."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cpu: bool) -> torch.device:
    """Return torch.device based on config."""
    if cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(p: Path) -> Path:
    """Create directory (parents=True) and return Path."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def has_labels_in_batch(batch: Dict) -> bool:
    """Return True if batch contains non-empty tensor under key 'label'."""
    y = batch.get("label", None)
    return isinstance(y, torch.Tensor) and y.numel() > 0
