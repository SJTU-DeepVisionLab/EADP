#!/usr/bin/env bash
# Shared public configuration for LLaVA-Video evaluation scripts.
#
# Required:
#   MODEL_PATH=/path/to/LLaVA-Video-7B-Qwen2
#
# Common optional overrides:
#   PYTHON=python
#   HF_HOME=$HOME/.cache/huggingface
#   DATASETS="mvbench longvideobench_val_v videomme"
#   GPU_IDS="0 1 2"
#   TOKENS="16 32 64"
#   MAX_FRAMES=64
#   CONV_TEMPLATE=qwen_1_5
#   CLIP_MODEL=openai/clip-vit-large-patch14-336
#   EADP_ALPHA=0.5
#   EADP_BETA=2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-}"
MAX_FRAMES="${MAX_FRAMES:-64}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-large-patch14-336}"
BATCH_SIZE="${BATCH_SIZE:-1}"

DATASETS="${DATASETS:-mvbench longvideobench_val_v videomme}"
TOKENS="${TOKENS:-16 32 64}"
GPU_IDS="${GPU_IDS:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_DIR}/outputs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -n "${HF_ENDPOINT:-}" ]; then
    export HF_ENDPOINT
fi

EVAL_CMD="${PYTHON} ${WORK_DIR}/eval/run_lmms_eval.py"

require_model_path() {
    if [ -z "${MODEL_PATH}" ]; then
        echo "ERROR: MODEL_PATH is required."
        echo "Example: MODEL_PATH=/path/to/LLaVA-Video-7B-Qwen2 bash $0"
        exit 1
    fi
}

prepare_output_dirs() {
    mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
}

print_common_config() {
    echo "Model: ${MODEL_PATH}"
    echo "Python: ${PYTHON}"
    echo "HF_HOME: ${HF_HOME}"
    echo "Max frames: ${MAX_FRAMES}"
    echo "Datasets: ${DATASETS}"
    echo "GPU IDs: ${GPU_IDS}"
    echo "Output root: ${OUTPUT_ROOT}"
}
