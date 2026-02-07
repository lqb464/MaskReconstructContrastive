from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ...training.utils import ensure_dir as _shared_ensure_dir

def ensure_dir(path: Path) -> None:
    _shared_ensure_dir(path)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
