#!/usr/bin/env bash
set -e
cd /mnt/data/itc_tts/src/piper/src/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
exec /mnt/data/itc_tts/venv/bin/python -m piper_train \
  --dataset-dir /mnt/data/itc_tts/train/pinyin/piper_dataset \
  --accelerator gpu --devices 1 --batch-size 32 --max-phoneme-ids 600 \
  --quality medium --max_epochs 400 --checkpoint-epochs 25 \
  --validation-split 0.02 --check_val_every_n_epoch 25 --num-test-examples 0 --precision 32 \
  --num_sanity_val_steps 0 --default_root_dir /mnt/data/itc_tts/train/runs/medium_pinyin
