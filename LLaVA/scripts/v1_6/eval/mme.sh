#!/bin/bash

CKPT_DIR="/path/to/models"
DATA_DIR="/path/to/playground/data/eval"

LLAVA_MODEL_SIZE=${LLAVA_MODEL_SIZE:-7b}
if [ "$LLAVA_MODEL_SIZE" = "13b" ]; then CKPT="llava-v1.6-vicuna-13b"; else CKPT="llava-v1.6-vicuna-7b"; fi
SPLIT="llava_mme"

TOKEN=${1}
BETA=${2:-1.0}
ALPHA=${3:-0.5}
PARAM="vtn_$((TOKEN * 5))_beta_${BETA}_alpha_${ALPHA}"

python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/MME/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --visual_token_num ${TOKEN} \
    --beta ${BETA} \
    --alpha ${ALPHA} \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py \
    --experiment ${SPLIT}/${CKPT}/${PARAM}

cd eval_tool

python calculation.py \
    --results_dir answers/${SPLIT}/${CKPT}/${PARAM} \
    --visual_token_num $((TOKEN * 5)) \
    --beta ${BETA} \
    --alpha ${ALPHA}
