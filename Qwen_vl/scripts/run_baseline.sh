#!/usr/bin/env bash
# Public baseline evaluation script for Qwen-VL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

prepare_output_dirs

OUTPUT_DIR="${OUTPUT_ROOT}/baseline"
mkdir -p "${OUTPUT_DIR}"

read -r -a DATASET_LIST <<< "${DATASETS}"
read -r -a GPU_LIST <<< "${GPU_IDS}"

run_task() {
    local gpu_id=$1
    local dataset_name=$2
    local tag="baseline_${MODEL_FAMILY}_${dataset_name}"

    echo "[$(date)] Starting ${tag} on GPU ${gpu_id}"
    run_vlmeval "${gpu_id}" "${BASELINE_MODEL}" "${dataset_name}" "${OUTPUT_DIR}" \
        > "${LOG_DIR}/${tag}.log" 2>&1
    local rc=$?
    echo "[$(date)] Finished ${tag} on GPU ${gpu_id}, exit code: ${rc}"
    return "${rc}"
}

echo "========================================="
echo "Baseline Evaluation"
print_common_config
echo "Model: ${BASELINE_MODEL}"
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
