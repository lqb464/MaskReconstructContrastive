#!/bin/bash
VER=$1
BASE_CH=$2
BATCH_SIZE=$3

if [ "$USE_MASKED_LOSS" = "true" ]; then
  MASKED_FLAG="--enable-masked-loss"
else
  MASKED_FLAG=""
fi

python unet/src/"$VER"/train.py --amp \
  --use-gn --use-se --use-multiscale \
  --pre-bias --pre-norm --pre-crop --pre-align \
  --data-source hf --base-ch "$BASE_CH" --batch-size "$BATCH_SIZE"\
  $MASKED_FLAG 
