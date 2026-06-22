#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

# ==============================
# Common Settings
# ==============================
export BASE_MODEL_PATH=gitted/Qwen3-VL-4B-Instruct
export BITS=16

EVAL_MODE=${EVAL_MODE:-covft}
if [ "$EVAL_MODE" = "base" ]; then
    CKPT_DIR=${BASE_MODEL_PATH}
    export TUNING_STRATEGY=auto
else
    CKPT_DIR=${CKPT_DIR:-checkpoints/qwen3vl-covft}
    export TUNING_STRATEGY=${TUNING_STRATEGY:-covft}
fi

export PYTHONPATH=src:CoVFT

# ==============================
# CoVFT Playground Benchmarks
# ==============================
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/eval_playground.sh mmvp $CKPT_DIR
echo "Done evaluation on mmvp, by using ${CKPT_DIR} model"

CUDA_VISIBLE_DEVICES=0 bash scripts/eval/eval_playground.sh realworldqa $CKPT_DIR
echo "Done evaluation on realworldqa, by using ${CKPT_DIR} model"

CUDA_VISIBLE_DEVICES=0 bash scripts/eval/eval_playground.sh coco $CKPT_DIR
echo "Done evaluation on coco, by using ${CKPT_DIR} model"

CUDA_VISIBLE_DEVICES=0 bash scripts/eval/eval_playground.sh ade $CKPT_DIR
echo "Done evaluation on ade, by using ${CKPT_DIR} model"

CUDA_VISIBLE_DEVICES=0 bash scripts/eval/eval_playground.sh omni $CKPT_DIR
echo "Done evaluation on omni, by using ${CKPT_DIR} model"

CUDA_VISIBLE_DEVICES=0 bash scripts/eval/eval_playground.sh ai2d $CKPT_DIR
echo "Done evaluation on ai2d, by using ${CKPT_DIR} model"
