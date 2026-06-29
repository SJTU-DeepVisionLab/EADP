#!/bin/bash

CKPT_DIR="/path/to/models"
DATA_DIR="/path/to/playground/data/eval"

LLAVA_MODEL_SIZE=${LLAVA_MODEL_SIZE:-7b}
if [ "$LLAVA_MODEL_SIZE" = "13b" ]; then CKPT="llava-v1.6-vicuna-13b"; else CKPT="llava-v1.6-vicuna-7b"; fi
SPLIT="llava_pope_test"

TOKEN=${1}
BETA=${2:-1.0}
ALPHA=${3:-1.0}

PARAM="vtn_$((TOKEN * 5))_beta_${BETA}_alpha_${ALPHA}"

python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --visual_token_num ${TOKEN} \
    --beta ${BETA} \
    --alpha ${ALPHA} \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ${DATA_DIR}/pope/coco \
    --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
    --result-file ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --visual_token_num $((TOKEN * 5)) \
    --beta ${BETA} \
    --alpha ${ALPHA}
