#!/usr/bin/env bash
# Shared public configuration for Qwen-VL evaluation scripts.
#
# Common examples:
#   MODEL_FAMILY=qwen2.5-7b DATASETS="MMBench_DEV_EN_V11 TextVQA_VAL" GPU_IDS="0 1" bash scripts/run_baseline.sh
#   MODEL_FAMILY=qwen3-8b TOKENS="128 256" EADP_ALPHA=0.5 EADP_BETA=2.0 bash scripts/run_eadp.sh
#
# Optional local model paths:
#   QWEN2_5_VL_3B_MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct
#   QWEN2_5_VL_7B_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
#   QWEN3_VL_8B_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLMEVALKIT_DIR="${WORK_DIR}/VLMEvalKit"
cd "${WORK_DIR}"

PYTHON="${PYTHON:-python}"
MODEL_FAMILY="${MODEL_FAMILY:-qwen2.5-7b}"
DATASETS="${DATASETS:-MMBench_DEV_EN_V11 TextVQA_VAL ChartQA_TEST AI2D_TEST OCRBench}"
TOKENS="${TOKENS:-128 256 512}"
GPU_IDS="${GPU_IDS:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_DIR}/outputs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
LMUData="${LMUData:-${HOME}/LMUData}"

export LMUData
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -n "${HF_ENDPOINT:-}" ]; then
    export HF_ENDPOINT
fi

case "${MODEL_FAMILY}" in
    qwen2.5-3b|qwen25-3b|qwen2_5_3b|qwen2_5_vl_3b)
        MODEL_PREFIX="Qwen2.5-VL-3B"
        BASELINE_MODEL="${MODEL_PREFIX}-Instruct-1008"
        export QWEN2_5_VL_3B_MODEL_PATH="${QWEN2_5_VL_3B_MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
        MODEL_PATH_DISPLAY="${QWEN2_5_VL_3B_MODEL_PATH}"
        ;;
    qwen2.5-7b|qwen25-7b|qwen2_5_7b|qwen2_5_vl_7b)
        MODEL_PREFIX="Qwen2.5-VL-7B"
        BASELINE_MODEL="${MODEL_PREFIX}-Instruct-1008"
        export QWEN2_5_VL_7B_MODEL_PATH="${QWEN2_5_VL_7B_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
        MODEL_PATH_DISPLAY="${QWEN2_5_VL_7B_MODEL_PATH}"
        ;;
    qwen3-8b|qwen3_8b|qwen3_vl_8b)
        MODEL_PREFIX="Qwen3-VL-8B"
        BASELINE_MODEL="${MODEL_PREFIX}-Instruct-1024"
        export QWEN3_VL_8B_MODEL_PATH="${QWEN3_VL_8B_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
        MODEL_PATH_DISPLAY="${QWEN3_VL_8B_MODEL_PATH}"
        ;;
    *)
        echo "ERROR: unsupported MODEL_FAMILY=${MODEL_FAMILY}" >&2
        echo "Supported: qwen2.5-3b, qwen2.5-7b, qwen3-8b" >&2
        exit 1
        ;;
esac

prepare_output_dirs() {
    mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
}

print_common_config() {
    echo "Model family: ${MODEL_FAMILY}"
    echo "Model path: ${MODEL_PATH_DISPLAY}"
    echo "Python: ${PYTHON}"
    echo "LMUData: ${LMUData}"
    echo "Datasets: ${DATASETS}"
    echo "GPU IDs: ${GPU_IDS}"
    echo "Output root: ${OUTPUT_ROOT}"
}

run_vlmeval() {
    local gpu_id=$1
    local model_name=$2
    local dataset_name=$3
    local output_dir=$4

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
        "${PYTHON}" "${VLMEVALKIT_DIR}/run.py" \
        --model "${model_name}" \
        --data "${dataset_name}" \
        --work-dir "${output_dir}" \
        --verbose
}
