#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-262144}"

bash "${SCRIPT_DIR}/launch.sh" src/train.py \
  --model_name_or_path gitted/Qwen3-VL-4B-Instruct \
  --data_path "/data/wxx/dataset/natural/llava_finetune/llava_v1_5_mix665k.json" \
  --image_folder "/data/wxx/dataset/natural/llava_finetune" \
  --image_max_pixels "${IMAGE_MAX_PIXELS}" \
  --skip_missing_images True \
  --output_dir checkpoints/qwen3vl-freeze-vision \
  --tuning_strategy freeze_vision \
  --bf16 True \
  --model_max_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0 \
  --lr_scheduler_type cosine \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 3 \
  --logging_steps 10 \
  --gradient_checkpointing True \
  --ddp_find_unused_parameters False \
  --remove_unused_columns False
