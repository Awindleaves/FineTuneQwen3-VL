#!/bin/bash
set -e

BENCHMARK=$1
CKPT_DIR=${2:-checkpoints/qwen3vl-covft}
CKPT_NAME=$(basename $CKPT_DIR)

BASE_MODEL_PATH=${BASE_MODEL_PATH:-gitted/Qwen3-VL-4B-Instruct}
EVAL_ROOT=${EVAL_ROOT:-playground/data/eval}
if [ "${CKPT_DIR}" = "${BASE_MODEL_PATH}" ]; then
    TUNING_STRATEGY=auto
else
    TUNING_STRATEGY=${TUNING_STRATEGY:-covft}
fi
BITS=${BITS:-16}

CONTEXT_EMBEDDING_MODEL=${CONTEXT_EMBEDDING_MODEL:-gitted/bert-base-uncased}

BENCHMARK_DIR=${EVAL_ROOT}/${BENCHMARK}
ANSWERS_FILE=${BENCHMARK_DIR}/answers/answers_${CKPT_NAME}.jsonl
INCORRECT_FILE=${BENCHMARK_DIR}/incorrect/incorrect_${CKPT_NAME}.jsonl
CSV_FILE=${BENCHMARK_DIR}/experiments.csv

mkdir -p ${BENCHMARK_DIR}/answers

python -m eval.model_playground_loader \
    --benchmark ${BENCHMARK} \
    --model-path ${CKPT_DIR} \
    --base-model-path ${BASE_MODEL_PATH} \
    --tuning-strategy ${TUNING_STRATEGY} \
    --bits ${BITS} \
    --context-embedding-model ${CONTEXT_EMBEDDING_MODEL} \
    --answers-file ${ANSWERS_FILE} \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --max-new-tokens 64

mv ${BENCHMARK_DIR}/answers/answers_${CKPT_NAME}_0.jsonl ${ANSWERS_FILE}

python ${BENCHMARK_DIR}/${BENCHMARK}_test.py \
    --answers_file ${ANSWERS_FILE} \
    --output_file ${INCORRECT_FILE} \
    --csv_file ${CSV_FILE}
