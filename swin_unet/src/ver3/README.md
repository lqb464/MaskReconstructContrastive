# ver3 Refactor Overview

This directory is the refactored `ver3` layout derived from `swin_unet/src/ver2` with backward-compatible entrypoints.

## Entrypoints

Legacy-compatible entry scripts:

- `python -m swin_unet.src.ver3.main`
- `python -m swin_unet.src.ver3.eval`
- `python -m swin_unet.src.ver3.alzheimer_classifier.main`
- `python -m swin_unet.src.ver3.mask_reconstruction.main`

Unified CLI:

- `python -m swin_unet.src.ver3.cli train-ssl ...`
- `python -m swin_unet.src.ver3.cli eval-ssl ...`
- `python -m swin_unet.src.ver3.cli train-cls ...`
- `python -m swin_unet.src.ver3.cli train-mask ...`

Compatibility aliases also exist in unified CLI:

- `train -> train-ssl`
- `eval -> eval-ssl`
- `cls -> train-cls`
- `mask -> train-mask`

## Folder Structure

`ver3/` layout:

- `main.py`, `eval.py`, `cli.py`
- `tasks/`
  - `ssl_reconstruction/`
  - `mask_reconstruction/`
  - `alzheimer_classifier/`
- `mask_reconstruction/` (compatibility shims)
- `alzheimer_classifier/` (compatibility shims)
- `common/`, `config/`, `data/`, `models/`, `training/`, `viz/`

Compatibility paths under `ver3/mask_reconstruction/*`, `ver3/alzheimer_classifier/*`, and `ver3/trainer.py` re-export task implementations from `ver3/tasks/*`.

## Migration Notes

- `ver2` remains untouched.
- Existing CLI flags remain available on the original ver3 entry scripts.
- New consolidated command surface is `ver3/cli.py` with subcommands.
- Internal task code now lives under `ver3/tasks/*`.

## Smoke Checks

Help checks:

- `python -m swin_unet.src.ver3.main --help`
- `python -m swin_unet.src.ver3.eval --help`
- `python -m swin_unet.src.ver3.alzheimer_classifier.main --help`
- `python -m swin_unet.src.ver3.mask_reconstruction.main --help`
- `python -m swin_unet.src.ver3.cli --help`

Minimal smoke scripts:

- `python scripts/smoke_ver3_imports.py`
- `bash scripts/smoke_ver3_cli_help.sh`
