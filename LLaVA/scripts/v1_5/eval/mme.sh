#!/bin/bash

CKPT_DIR="/path/to/models"
DATA_DIR="/path/to/playground/data/eval"

LLAVA_MODEL_SIZE=${LLAVA_MODEL_SIZE:-7b}
if [ "$LLAVA_MODEL_SIZE" = "13b" ]; then CKPT="llava-v1.5-13b"; else CKPT="llava-v1.5-7b"; fi
SPLIT="llava_mme"

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
    --visual_token_num ${TOKEN} \
    --beta ${BETA} \
    --alpha ${ALPHA}
