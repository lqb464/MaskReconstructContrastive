#!/bin/bash
VER=$1
ADNI_PROC_PATH=$2
BASE_CH=$3

if [ "$USE_MASKED_LOSS" = "true" ]; then
  MASKED_FLAG="--enable-masked-loss"
else
  MASKED_FLAG=""
fi

python unet/src/"$VER"/train.py --amp \
  --use-gn --use-se --use-multiscale \
  --pre-bias --pre-norm --pre-crop --pre-align \
  --data-source adni_preproc --adni-path "$ADNI_PROC_PATH" --image-type "$IMAGE_TYPE" \
  --base-ch BASH_CH \
  $MASKED_FLAG 
