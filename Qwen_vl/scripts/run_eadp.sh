#!/usr/bin/env bash
# Public EADP evaluation script for Qwen-VL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

EADP_ALPHA="${EADP_ALPHA:-0.5}"
EADP_BETA="${EADP_BETA:-2.0}"

prepare_output_dirs

OUTPUT_DIR="${OUTPUT_ROOT}/eadp"
mkdir -p "${OUTPUT_DIR}"

read -r -a DATASET_LIST <<< "${DATASETS}"
read -r -a TOKEN_LIST <<< "${TOKENS}"
read -r -a GPU_LIST <<< "${GPU_IDS}"

run_task() {
    local gpu_id=$1
    local token_num=$2
    local dataset_name=$3
    local model_name="${MODEL_PREFIX}-EADP-${token_num}-a${EADP_ALPHA}-b${EADP_BETA}"
    local tag="eadp_${MODEL_FAMILY}_t${token_num}_a${EADP_ALPHA}_b${EADP_BETA}_${dataset_name}"

    echo "[$(date)] Starting ${tag} on GPU ${gpu_id}"
    run_vlmeval "${gpu_id}" "${model_name}" "${dataset_name}" "${OUTPUT_DIR}" \
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
echo "========================================="

pids=()
job_idx=0
for token_num in "${TOKEN_LIST[@]}"; do
    for dataset in "${DATASET_LIST[@]}"; do
        gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"
        run_task "${gpu}" "${token_num}" "${dataset}" &
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
