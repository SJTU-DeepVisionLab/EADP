#!/bin/bash

CKPT_DIR="/path/to/models"
DATA_DIR="/path/to/playground/data/eval"

LLAVA_MODEL_SIZE=${LLAVA_MODEL_SIZE:-7b}
if [ "$LLAVA_MODEL_SIZE" = "13b" ]; then CKPT="llava-v1.5-13b"; else CKPT="llava-v1.5-7b"; fi
SPLIT="llava_textvqa_val_v051_ocr"

TOKEN=${1}
BETA=${2:-1.0}
ALPHA=${3:-0.5}
ENTROPY="${4:-}"
if [ -n "$ENTROPY" ]; then
  export ENTROPY_KEEP_RATIO="${ENTROPY}"
  PARAM="vtn_${TOKEN}_beta_${BETA}_alpha_${ALPHA}_entropy_${ENTROPY}"
else
  PARAM="vtn_${TOKEN}_beta_${BETA}_alpha_${ALPHA}"
fi
SMOOTHING="${5:-}"
if [ -n "$SMOOTHING" ]; then
  PARAM="${PARAM}_smoothing_${SMOOTHING}"
fi
LOCAL_AGG="${6:-}"
if [ -n "$LOCAL_AGG" ]; then
  PARAM="${PARAM}_local_agg_${LOCAL_AGG}"
fi
TEXT_AGG="${7:-}"
if [ -n "$TEXT_AGG" ]; then
  PARAM="${PARAM}_text_agg_${TEXT_AGG}"
fi
PARAM_SUFFIX="${8:-}"
if [ -n "$PARAM_SUFFIX" ]; then
  PARAM="${PARAM}_${PARAM_SUFFIX}"
fi

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
    --visual_token_num ${TOKEN} \
    --beta ${BETA} \
    --alpha ${ALPHA}
