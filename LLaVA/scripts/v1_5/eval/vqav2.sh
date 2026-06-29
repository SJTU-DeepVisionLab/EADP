#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# 13B needs both GPUs in one process; do not split into per-GPU workers
if [ "${LLAVA_MODEL_SIZE:-7b}" = "13b" ]; then CHUNKS=1; else CHUNKS=${#GPULIST[@]}; fi

CKPT_DIR="/path/to/models"
DATA_DIR="/path/to/playground/data/eval"

LLAVA_MODEL_SIZE=${LLAVA_MODEL_SIZE:-7b}
if [ "$LLAVA_MODEL_SIZE" = "13b" ]; then CKPT="llava-v1.5-13b"; else CKPT="llava-v1.5-7b"; fi
SPLIT="llava_vqav2_mscoco_test-dev2015"

TOKEN=${1}
BETA=${2:-1.0}
ALPHA=${3:-0.5}
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
PARAM="vtn_${TOKEN}_beta_${BETA}_alpha_${ALPHA}_time_${TIMESTAMP}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    if [ "$CHUNKS" -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES="$gpu_list"
    else
        export CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}
    fi
    python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/vqav2/${SPLIT}.jsonl \
        --image-folder ${DATA_DIR}/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/${SPLIT}/${CKPT}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --visual_token_num ${TOKEN} \
        --beta ${BETA} \
        --alpha ${ALPHA} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

VQAV2_DIR="./playground/data/eval/vqav2"
output_file=./playground/data/eval/vqav2/answers/${SPLIT}/${CKPT}/${PARAM}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/${SPLIT}/${CKPT}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py \
    --dir ${VQAV2_DIR} \
    --src answers/${SPLIT}/${CKPT}/${PARAM}/merge.jsonl \
    --dst answers_upload/${SPLIT}/${CKPT}/${PARAM}.json
