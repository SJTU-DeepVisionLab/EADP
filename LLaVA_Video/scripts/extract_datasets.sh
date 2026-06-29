#!/usr/bin/env bash
# Extract downloaded video benchmark archives.
#
# Usage:
#   HF_HOME=$HOME/.cache/huggingface bash scripts/extract_datasets.sh
#   bash scripts/extract_datasets.sh videomme

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

TARGET="${1:-all}"
EXTRACT_LOG="${EXTRACT_LOG:-${HF_HOME}/llava_video_extract.log}"
mkdir -p "${HF_HOME}"

extract_videomme() {
    local dir="${HF_HOME}/videomme"
    echo ""
    echo ">>> Extracting Video-MME"
    if [ ! -d "${dir}" ]; then
        echo "  SKIP: ${dir} does not exist."
        return
    fi
    cd "${dir}"
    mkdir -p data
    for zipfile in videos_chunked_*.zip; do
        if [ -f "${zipfile}" ]; then
            echo "  Extracting ${zipfile}"
            unzip -o "${zipfile}" -d . >> "${EXTRACT_LOG}" 2>&1
        fi
    done
    if [ -d "videos" ] && [ ! -d "data" ]; then
        mv videos data
    fi
    if [ -f "subtitle.zip" ] && [ ! -d "subtitle" ]; then
        echo "  Extracting subtitle.zip"
        unzip -o subtitle.zip -d . >> "${EXTRACT_LOG}" 2>&1
    fi
    local count
    count=$(find data -type f \( -name "*.mp4" -o -name "*.MP4" \) 2>/dev/null | wc -l)
    echo "  Video-MME videos: ${count}"
}

extract_longvideobench() {
    local dir="${HF_HOME}/longvideobench"
    echo ""
    echo ">>> Extracting LongVideoBench"
    if [ ! -d "${dir}" ]; then
        echo "  SKIP: ${dir} does not exist."
        return
    fi
    cd "${dir}"
    if [ -f "videos.tar.part.aa" ] && [ ! -d "videos" ]; then
        echo "  Concatenating and extracting videos.tar.part.*"
        cat videos.tar.part.* | tar xf -
    fi
    if [ -f "subtitles.tar" ] && [ ! -d "subtitles" ]; then
        echo "  Extracting subtitles.tar"
        tar xf subtitles.tar
    fi
    local count
    count=$(find videos -type f -name "*.mp4" 2>/dev/null | wc -l)
    echo "  LongVideoBench videos: ${count}"
}

check_mvbench() {
    local dir="${HF_HOME}/mvbench_video"
    echo ""
    echo ">>> Checking MVBench"
    if [ ! -d "${dir}" ]; then
        echo "  SKIP: ${dir} does not exist."
        return
    fi
    local count
    count=$(find "${dir}" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.webm" \) 2>/dev/null | wc -l)
    echo "  MVBench videos: ${count}"
}

echo "========================================="
echo "Extracting video datasets"
echo "HF_HOME: ${HF_HOME}"
echo "Target: ${TARGET}"
echo "Started: $(date)"
echo "========================================="

case "${TARGET}" in
    all)
        extract_videomme
        extract_longvideobench
        check_mvbench
        ;;
    videomme)
        extract_videomme
        ;;
    longvideobench)
        extract_longvideobench
        ;;
    mvbench)
        check_mvbench
        ;;
    *)
        echo "Unknown dataset: ${TARGET}"
        echo "Expected: all, mvbench, longvideobench, videomme"
        exit 1
        ;;
esac

echo ""
echo "Dataset directory summary:"
for dir in mvbench_video longvideobench videomme; do
    if [ -d "${HF_HOME}/${dir}" ]; then
        size=$(du -sh "${HF_HOME}/${dir}" 2>/dev/null | cut -f1)
        vids=$(find "${HF_HOME}/${dir}" -type f \( -name "*.mp4" -o -name "*.MP4" -o -name "*.avi" -o -name "*.webm" \) 2>/dev/null | wc -l)
        echo "  ${dir}: ${size}, ${vids} video files"
    fi
done

echo "Completed: $(date)"
echo "========================================="
