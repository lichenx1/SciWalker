#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
SCIDATA_PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SCIDATA_PROJECT_DIR"
exec "${SCIDATA_PYTHON:-python3}" -m scidata.workflow generate \
    --config configs/llm.local.json \
    --pipeline-config configs/pipeline.local.json "$@"
