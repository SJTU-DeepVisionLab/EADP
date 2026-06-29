#!/bin/bash

CKPT_DIR="/path/to/models"
DATA_DIR="/path/to/playground/data/eval"

LLAVA_MODEL_SIZE=${LLAVA_MODEL_SIZE:-7b}
if [ "$LLAVA_MODEL_SIZE" = "13b" ]; then CKPT="llava-v1.6-vicuna-13b"; else CKPT="llava-v1.6-vicuna-7b"; fi
SPLIT="llava_test"

TOKEN=${1}
BETA=${2:-1.0}
ALPHA=${3:-0.5}
PARAM="vtn_$((TOKEN * 5))_beta_${BETA}_alpha_${ALPHA}"

python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/vizwiz/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --visual_token_num ${TOKEN} \
    --beta ${BETA} \
    --alpha ${ALPHA} \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/${SPLIT}.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${SPLIT}/${CKPT}/${PARAM}.json
