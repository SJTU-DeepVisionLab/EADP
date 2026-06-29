#!/usr/bin/env bash
# Public baseline evaluation script for LLaVA-Video.
#
# Example:
#   MODEL_PATH=/path/to/LLaVA-Video-7B-Qwen2 GPU_IDS="0 1 2" bash scripts/run_baseline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

require_model_path
prepare_output_dirs

OUTPUT_DIR="${OUTPUT_ROOT}/baseline"
mkdir -p "${OUTPUT_DIR}"

read -r -a DATASET_LIST <<< "${DATASETS}"
read -r -a GPU_LIST <<< "${GPU_IDS}"

run_task() {
    local gpu_id=$1
    local task_name=$2
    local output_name="${task_name/longvideobench_val_v/longvideobench}"
    local tag="baseline_${output_name}"

    echo "[$(date)] Starting ${tag} on GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        "${PYTHON}" "${WORK_DIR}/eval/run_lmms_eval.py" \
        --model llava_vid \
        --model_args "pretrained=${MODEL_PATH},max_frames_num=${MAX_FRAMES},conv_template=${CONV_TEMPLATE},video_decode_backend=decord,mm_spatial_pool_mode=average" \
        --tasks "${task_name}" \
        --batch_size "${BATCH_SIZE}" \
        --log_samples \
        --log_samples_suffix "baseline" \
        --output_path "${OUTPUT_DIR}/${output_name}" \
        > "${LOG_DIR}/${tag}.log" 2>&1
    local rc=$?
    echo "[$(date)] Finished ${tag} on GPU ${gpu_id}, exit code: ${rc}"
    return "${rc}"
}

echo "========================================="
echo "Baseline Evaluation"
print_common_config
echo "========================================="

pids=()
job_idx=0
for dataset in "${DATASET_LIST[@]}"; do
    gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"
    run_task "${gpu}" "${dataset}" &
    pids+=("$!")
    job_idx=$((job_idx + 1))
    if [ "${#pids[@]}" -ge "${#GPU_LIST[@]}" ]; then
        wait "${pids[@]}"
        pids=()
    fi
done

if [ "${#pids[@]}" -gt 0 ]; then
    wait "${pids[@]}"
fi

echo "========================================="
echo "[$(date)] Baseline evaluation complete."
echo "========================================="
