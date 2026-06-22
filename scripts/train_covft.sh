#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_EMBEDDING_MODEL="${CONTEXT_EMBEDDING_MODEL:-gitted/bert-base-uncased}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-262144}"
COVFT_MOE_NUM_EXPERTS="${COVFT_MOE_NUM_EXPERTS:-2}"
COVFT_MOE_TOP_K="${COVFT_MOE_TOP_K:-2}"
COVFT_MOE_START_LAYER="${COVFT_MOE_START_LAYER:--2}"
COVFT_MODEL_MAX_LENGTH="${COVFT_MODEL_MAX_LENGTH:-1024}"
COVFT_IMAGE_MAX_PIXELS="${COVFT_IMAGE_MAX_PIXELS:-${IMAGE_MAX_PIXELS}}"
COVFT_GRADIENT_ACCUMULATION_STEPS="${COVFT_GRADIENT_ACCUMULATION_STEPS:-16}"
COVFT_TRAIN_LLM="${COVFT_TRAIN_LLM:-False}"
COVFT_LLM_LORA="${COVFT_LLM_LORA:-True}"

bash "${SCRIPT_DIR}/launch.sh" src/train.py \
  --model_name_or_path gitted/Qwen3-VL-4B-Instruct \
  --data_path "/data/wxx/dataset/natural/llava_finetune/llava_v1_5_mix665k.json" \
  --image_folder "/data/wxx/dataset/natural/llava_finetune" \
  --image_max_pixels "${COVFT_IMAGE_MAX_PIXELS}" \
  --skip_missing_images True \
  --output_dir checkpoints/qwen3vl-covft \
  --tuning_strategy covft \
  --moe_num_experts "${COVFT_MOE_NUM_EXPERTS}" \
  --moe_top_k "${COVFT_MOE_TOP_K}" \
  --moe_start_layer "${COVFT_MOE_START_LAYER}" \
  --context_embedding_model "${CONTEXT_EMBEDDING_MODEL}" \
  --context_embedding_max_length 256 \
  --covft_attn_heads 4 \
  --covft_train_llm "${COVFT_TRAIN_LLM}" \
  --covft_llm_lora "${COVFT_LLM_LORA}" \
  --bf16 True \
  --model_max_length "${COVFT_MODEL_MAX_LENGTH}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${COVFT_GRADIENT_ACCUMULATION_STEPS}" \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0 \
  --lr_scheduler_type cosine \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 3 \
  --logging_steps 10 \
  --gradient_checkpointing False \
  --ddp_find_unused_parameters True \
  --remove_unused_columns False
