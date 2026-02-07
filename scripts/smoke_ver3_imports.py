#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_IMPORTS = [
    "swin_unet.src.ver3",
    "swin_unet.src.ver3.main",
    "swin_unet.src.ver3.eval",
    "swin_unet.src.ver3.cli",
    "swin_unet.src.ver3.alzheimer_classifier.main",
    "swin_unet.src.ver3.mask_reconstruction.main",
    "swin_unet.src.ver3.tasks.alzheimer_classifier.main",
    "swin_unet.src.ver3.tasks.mask_reconstruction.main",
]

TORCH_IMPORTS = [
    "swin_unet.src.ver3.tasks.ssl_reconstruction.trainer",
    "swin_unet.src.ver3.tasks.mask_reconstruction.trainer",
    "swin_unet.src.ver3.tasks.alzheimer_classifier.train",
]


def try_import(mod_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(mod_name)
        return True, "OK"
    except ModuleNotFoundError as exc:
        return False, f"MISSING: {exc.name}"
    except Exception as exc:  # pragma: no cover - smoke path
        return False, f"ERROR: {exc.__class__.__name__}: {exc}"


def main() -> int:
    failures = []

    for mod in BASE_IMPORTS:
        ok, msg = try_import(mod)
        print(f"[BASE] {mod}: {msg}")
        if not ok:
            failures.append(mod)

    torch_available, _ = try_import("torch")
    if torch_available:
        for mod in TORCH_IMPORTS:
            ok, msg = try_import(mod)
            print(f"[TORCH] {mod}: {msg}")
            if not ok:
                failures.append(mod)
    else:
        print("[TORCH] torch not installed; skipping torch-dependent imports")

    if failures:
        print("SMOKE_FAIL")
        return 1

    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
