#!/bin/bash

CKPT_DIR="/path/to/models"
DATA_DIR="/path/to/playground/data/eval"

LLAVA_MODEL_SIZE=${LLAVA_MODEL_SIZE:-7b}
if [ "$LLAVA_MODEL_SIZE" = "13b" ]; then CKPT="llava-v1.6-vicuna-13b"; else CKPT="llava-v1.6-vicuna-7b"; fi
SPLIT="llava_textvqa_val_v051_ocr"

TOKEN=${1}
BETA=${2:-1.0}
ALPHA=${3:-0.5}
PARAM="vtn_$((TOKEN * 5))_beta_${BETA}_alpha_${ALPHA}"

python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/textvqa/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --visual_token_num ${TOKEN} \
    --beta ${BETA} \
    --alpha ${ALPHA} \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ${DATA_DIR}/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --visual_token_num $((TOKEN * 5)) \
    --beta ${BETA} \
    --alpha ${ALPHA}
