#!/bin/bash

CKPT_DIR="/path/to/models"
DATA_DIR="/path/to/playground/data/eval"

LLAVA_MODEL_SIZE=${LLAVA_MODEL_SIZE:-7b}
if [ "$LLAVA_MODEL_SIZE" = "13b" ]; then CKPT="llava-v1.6-vicuna-13b"; else CKPT="llava-v1.6-vicuna-7b"; fi
SPLIT="llava_test_CQM-I"

TOKEN=${1}
BETA=${2:-1.0}
ALPHA=${3:-0.5}
PARAM="vtn_$((TOKEN * 5))_beta_${BETA}_alpha_${ALPHA}"

python -m llava.eval.model_vqa_science \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/scienceqa/${SPLIT}.json \
    --image-folder ${DATA_DIR}/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --visual_token_num ${TOKEN} \
    --beta ${BETA} \
    --alpha ${ALPHA} \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_science_qa \
    --base-dir ${DATA_DIR}/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/${PARAM}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/${PARAM}_result.json \
    --visual_token_num $((TOKEN * 5)) \
    --beta ${BETA} \
    --alpha ${ALPHA}
