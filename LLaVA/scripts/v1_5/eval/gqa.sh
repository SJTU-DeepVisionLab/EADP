#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# 13B needs both GPUs in one process; do not split into per-GPU workers
if [ "${LLAVA_MODEL_SIZE:-7b}" = "13b" ]; then CHUNKS=1; else CHUNKS=${#GPULIST[@]}; fi

CKPT_DIR="/path/to/models"
DATA_DIR="/path/to/playground/data/eval"

LLAVA_MODEL_SIZE=${LLAVA_MODEL_SIZE:-7b}
if [ "$LLAVA_MODEL_SIZE" = "13b" ]; then CKPT="llava-v1.5-13b"; else CKPT="llava-v1.5-7b"; fi
SPLIT="llava_gqa_testdev_balanced"

TOKEN=${1}
BETA=${2:-1.0}
ALPHA=${3:-0.5}
PARAM="vtn_${TOKEN}_beta_${BETA}_alpha_${ALPHA}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    if [ "$CHUNKS" -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES="$gpu_list"
    else
        export CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}
    fi
    python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/gqa/${SPLIT}.jsonl \
        --image-folder ${DATA_DIR}/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --visual_token_num ${TOKEN} \
        --beta ${BETA} \
        --alpha ${ALPHA} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

GQA_DIR="./playground/data/eval/gqa/data"
output_file=./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/${PARAM}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst ${GQA_DIR}/testdev_balanced_predictions.json

if [ ! -f "${GQA_DIR}/testdev_balanced_questions.json" ]; then
    echo "Warning: ${GQA_DIR}/testdev_balanced_questions.json not found."
    echo "Please run scripts/v1_5/eval/download_gqa_data.sh to download the GQA questions data."
fi

cd ${GQA_DIR}
python eval/eval.py \
    --tier testdev_balanced \
    --visual_token_num ${TOKEN} \
    --beta ${BETA} \
    --alpha ${ALPHA}

rm testdev_balanced_predictions.json
