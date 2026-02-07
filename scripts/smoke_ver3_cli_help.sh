#!/usr/bin/env bash
set -euo pipefail

python -m swin_unet.src.ver3.main --help >/dev/null
python -m swin_unet.src.ver3.eval --help >/dev/null
python -m swin_unet.src.ver3.alzheimer_classifier.main --help >/dev/null
python -m swin_unet.src.ver3.mask_reconstruction.main --help >/dev/null
python -m swin_unet.src.ver3.cli --help >/dev/null

echo "SMOKE_OK"
