#!/usr/bin/env bash
# Download the public video benchmarks used in LLaVA-Video experiments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

DATASETS_DOWNLOAD="${DATASETS_DOWNLOAD:-mvbench longvideobench videomme}"
DOWNLOAD_LOG="${DOWNLOAD_LOG:-${HF_HOME}/llava_video_download.log}"

mkdir -p "${HF_HOME}"

echo "========================================="
echo "Dataset download started: $(date)"
echo "HF_HOME: ${HF_HOME}"
echo "Datasets: ${DATASETS_DOWNLOAD}"
echo "Log: ${DOWNLOAD_LOG}"
echo "========================================="

exec > >(tee -a "${DOWNLOAD_LOG}") 2>&1

read -r -a DOWNLOAD_LIST <<< "${DATASETS_DOWNLOAD}"
"${PYTHON}" "${SCRIPT_DIR}/download_datasets.py" --hf-home "${HF_HOME}" --datasets "${DOWNLOAD_LIST[@]}"

echo ""
echo "Dataset cache summary:"
for dir in mvbench_video longvideobench videomme; do
    if [ -d "${HF_HOME}/${dir}" ]; then
        size=$(du -sh "${HF_HOME}/${dir}" 2>/dev/null | cut -f1)
        echo "  ${dir}: ${size}"
    else
        echo "  ${dir}: NOT FOUND"
    fi
done

echo "========================================="
echo "Download completed: $(date)"
echo "========================================="
