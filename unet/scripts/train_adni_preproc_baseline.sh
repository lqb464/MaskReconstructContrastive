#!/bin/bash
VER=$1
ADNI_PROC_PATH=$2
BASE_CH=$3
BATCH_SIZE=$4


if [ "$USE_MASKED_LOSS" = "true" ]; then
  MASKED_FLAG="--enable-masked-loss"
else
  MASKED_FLAG=""
fi

python unet/src/"$VER"/train.py --amp \
  --use-gn --use-se --use-multiscale \
  --pre-bias --pre-norm --pre-crop --pre-align \
  --data-source adni_preproc --adni-preproc-path "$ADNI_PROC_PATH" \
  --base-ch "$BASE_CH" --batch-size "$BATCH_SIZE"\
  $MASKED_FLAG 
