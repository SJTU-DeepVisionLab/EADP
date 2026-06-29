#!/bin/bash

CKPT_DIR="/path/to/models"
EVAL_DIR="./playground/data/eval/vizwiz"
LLAVA_MODEL_SIZE=${LLAVA_MODEL_SIZE:-7b}
if [ "$LLAVA_MODEL_SIZE" = "13b" ]; then CKPT="llava-v1.5-13b"; else CKPT="llava-v1.5-7b"; fi

# Arguments
TOKEN=${1:-"original"} # Default to original if not provided
BETA=${2:-1.0}
ALPHA=${3:-0.0}
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

# Paths
ANNOTATION_FILE="${EVAL_DIR}/val.json"
QUESTION_FILE="${EVAL_DIR}/llava_vizwiz_val.jsonl"
IMAGE_FOLDER="${EVAL_DIR}/val"
ANSWERS_FILE="${EVAL_DIR}/answers/${CKPT}/${PARAM}.jsonl"

# 1. Prepare Question File
if [ ! -f "$QUESTION_FILE" ]; then
    echo "Question file $QUESTION_FILE not found."
    if [ -f "$ANNOTATION_FILE" ]; then
        echo "Converting $ANNOTATION_FILE to $QUESTION_FILE..."
        python scripts/convert_vizwiz_val_to_llava.py \
            --src "$ANNOTATION_FILE" \
            --dst "$QUESTION_FILE"
    else
        echo "Error: Annotation file $ANNOTATION_FILE not found. Please download VizWiz val.json."
        exit 1
    fi
fi

# 2. Inference
echo "Running inference..."
python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ${QUESTION_FILE} \
    --image-folder ${IMAGE_FOLDER} \
    --answers-file ${ANSWERS_FILE} \
    --visual_token_num ${TOKEN} \
    --beta ${BETA} \
    --alpha ${ALPHA} \
    --temperature 0 \
    --conv-mode vicuna_v1

# 3. Evaluation
echo "Running evaluation..."
python scripts/eval_vizwiz.py \
    --annotation-file ${ANNOTATION_FILE} \
    --result-file ${ANSWERS_FILE} \
    --visual_token_num ${TOKEN} \
    --beta ${BETA} \
    --alpha ${ALPHA}
