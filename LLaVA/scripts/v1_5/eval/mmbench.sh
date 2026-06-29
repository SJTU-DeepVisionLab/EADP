#!/bin/bash

CKPT_DIR="/path/to/models"
DATA_DIR="/path/to/playground/data/eval"

LLAVA_MODEL_SIZE=${LLAVA_MODEL_SIZE:-7b}
if [ "$LLAVA_MODEL_SIZE" = "13b" ]; then CKPT="llava-v1.5-13b"; else CKPT="llava-v1.5-7b"; fi
SPLIT="mmbench_dev_20230712"

TOKEN=${1}
BETA=${2:-1.0}
ALPHA=${3:-0.5}
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
PARAM="vtn_${TOKEN}_beta_${BETA}_alpha_${ALPHA}_time_${TIMESTAMP}"

python -m llava.eval.model_vqa_mmbench \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ${DATA_DIR}/mmbench/${SPLIT}.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --visual_token_num ${TOKEN} \
    --beta ${BETA} \
    --alpha ${ALPHA} \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/${SPLIT}/${CKPT}

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${DATA_DIR}/mmbench/${SPLIT}.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/${SPLIT}/${CKPT} \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/${SPLIT}/${CKPT} \
    --experiment ${PARAM}
