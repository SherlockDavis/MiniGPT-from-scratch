#!/usr/bin/env bash
# Download TinyStoriesV2-GPT4 train + validation files from the Hugging Face Hub.
#
# Files land in ./data/ relative to the project root. curl --continue-at - makes
# this script safe to re-run after a partial download.
#
# Note: the train file is ~2 GB. Use `MINIGPT_VALID_ONLY=1 bash scripts/download_data.sh`
# to fetch only the validation file (useful for smoke tests).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
mkdir -p "${DATA_DIR}"

BASE_URL="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"

download() {
    local fname="$1"
    local url="${BASE_URL}/${fname}"
    local dest="${DATA_DIR}/${fname}"
    echo ">>> ${fname}"
    curl --location --fail --continue-at - --output "${dest}" "${url}"
}

download "TinyStoriesV2-GPT4-valid.txt"

if [[ "${MINIGPT_VALID_ONLY:-0}" != "1" ]]; then
    download "TinyStoriesV2-GPT4-train.txt"
else
    echo "MINIGPT_VALID_ONLY=1 set — skipping train file."
fi

echo "Done. Files in ${DATA_DIR}:"
ls -lh "${DATA_DIR}"
