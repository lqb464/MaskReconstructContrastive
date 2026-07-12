from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch

from ..training.utils import set_seed as _base_set_seed

def set_seed(
    seed: int,
    *,
    deterministic: bool | None = None,
    benchmark: bool | None = None,
) -> None:
    """
    Seed helper for skull_stripping.
    Default behavior matches base set_seed unless optional cudnn flags are provided.
    """
    _base_set_seed(int(seed))
    if deterministic is not None:
        torch.backends.cudnn.deterministic = bool(deterministic)
    if benchmark is not None:
        torch.backends.cudnn.benchmark = bool(benchmark)

def make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    base_seed = int(base_seed)

    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = (base_seed + int(worker_id)) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _worker_init_fn

__all__ = ["set_seed", "make_worker_init_fn"]
