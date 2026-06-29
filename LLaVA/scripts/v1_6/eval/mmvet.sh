#!/bin/bash

CKPT_DIR="/path/to/models"
DATA_DIR="/path/to/playground/data/eval"

LLAVA_MODEL_SIZE=${LLAVA_MODEL_SIZE:-7b}
if [ "$LLAVA_MODEL_SIZE" = "13b" ]; then CKPT="llava-v1.6-vicuna-13b"; else CKPT="llava-v1.6-vicuna-7b"; fi
SPLIT="llava-mm-vet"

TOKEN=${1}
BETA=${2:-1.0}
ALPHA=${3:-0.5}
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
PARAM="vtn_$((TOKEN * 5))_beta_${BETA}_alpha_${ALPHA}_time_${TIMESTAMP}"

python -m llava.eval.model_vqa \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/mm-vet/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --visual_token_num ${TOKEN} \
    --beta ${BETA} \
    --alpha ${ALPHA} \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/answers_upload/${SPLIT}/${CKPT}

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --dst ./playground/data/eval/mm-vet/answers_upload/${SPLIT}/${CKPT}/${PARAM}.json
