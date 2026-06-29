#!/usr/bin/env bash
# Public EADP evaluation script for LLaVA-Video.
#
# Example:
#   MODEL_PATH=/path/to/LLaVA-Video-7B-Qwen2 TOKENS="16 32 64" EADP_ALPHA=0.5 EADP_BETA=2.0 GPU_IDS="0 1 2" bash scripts/run_eadp.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

EADP_ALPHA="${EADP_ALPHA:-0.5}"
EADP_BETA="${EADP_BETA:-2.0}"

require_model_path
prepare_output_dirs

OUTPUT_DIR="${OUTPUT_ROOT}/eadp"
mkdir -p "${OUTPUT_DIR}"

read -r -a DATASET_LIST <<< "${DATASETS}"
read -r -a TOKEN_LIST <<< "${TOKENS}"
read -r -a GPU_LIST <<< "${GPU_IDS}"

run_single() {
    local gpu_id=$1
    local token_num=$2
    local task_name=$3
    local tag="eadp_t${token_num}_a${EADP_ALPHA}_b${EADP_BETA}_${task_name}"

    echo "[$(date)] Starting ${tag} on GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        "${PYTHON}" "${WORK_DIR}/eval/run_lmms_eval.py" \
        --model llava_vid_pruned \
        --model_args "pretrained=${MODEL_PATH},max_frames_num=${MAX_FRAMES},conv_template=${CONV_TEMPLATE},video_decode_backend=decord,mm_spatial_pool_mode=average,pruner_type=eadp,visual_token_num=${token_num},alpha=${EADP_ALPHA},beta=${EADP_BETA},clip_model_name=${CLIP_MODEL}" \
        --tasks "${task_name}" \
        --batch_size "${BATCH_SIZE}" \
        --log_samples \
        --log_samples_suffix "eadp_t${token_num}_a${EADP_ALPHA}_b${EADP_BETA}" \
        --output_path "${OUTPUT_DIR}/t${token_num}_a${EADP_ALPHA}_b${EADP_BETA}/${task_name}" \
        > "${LOG_DIR}/${tag}.log" 2>&1
    local rc=$?
    echo "[$(date)] Finished ${tag} on GPU ${gpu_id}, exit code: ${rc}"
    return "${rc}"
}

echo "========================================="
echo "EADP Evaluation"
print_common_config
echo "Token counts: ${TOKENS}"
echo "Alpha: ${EADP_ALPHA}"
echo "Beta: ${EADP_BETA}"
echo "CLIP model: ${CLIP_MODEL}"
echo "========================================="

pids=()
job_idx=0
for token_num in "${TOKEN_LIST[@]}"; do
    for dataset in "${DATASET_LIST[@]}"; do
        gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"
        run_single "${gpu}" "${token_num}" "${dataset}" &
        pids+=("$!")
        job_idx=$((job_idx + 1))
        if [ "${#pids[@]}" -ge "${#GPU_LIST[@]}" ]; then
            wait "${pids[@]}"
            pids=()
        fi
    done
done

if [ "${#pids[@]}" -gt 0 ]; then
    wait "${pids[@]}"
fi

echo "========================================="
echo "[$(date)] EADP evaluation complete."
echo "========================================="
