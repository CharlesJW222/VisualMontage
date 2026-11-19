#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
DATASET="${MONTAGE_DATASET:-Video_Dataset_Categories/01_Emergency_and_Disaster_Response/air_ground_firefighting}"
OUTPUT="${MONTAGE_OUTPUT:-attack_plan/x59_simulator_to_first_flight_local_round}"
LOCAL_BASE="${MONTAGE_LOCAL_API_BASE:-http://127.0.0.1:8081/v1}"
LOCAL_KEY_ENV="${MONTAGE_LOCAL_API_KEY_ENV:-MONTAGE_LOCAL_API_KEY}"
LOCAL_MODEL="${MONTAGE_LOCAL_MODEL_NAME:-$PROJECT_ROOT/models/qwen3-vl-8b-instruct-4bit}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python virtual environment not found: $PYTHON_BIN" >&2
  exit 2
fi

export MONTAGE_LOCAL_API_KEY="${MONTAGE_LOCAL_API_KEY:-your key here}"

EXTRA_ARGS=("$@")
if [[ " ${EXTRA_ARGS[*]} " == *" --enable-victim-validation "* ]]; then
  if [[ " ${EXTRA_ARGS[*]} " != *" --victim-model "* && " ${EXTRA_ARGS[*]} " != *" --victim-models "* ]]; then
    EXTRA_ARGS+=(--victim-model "$LOCAL_MODEL")
  fi
  if [[ " ${EXTRA_ARGS[*]} " != *" --judge-model "* ]]; then
    EXTRA_ARGS+=(--judge-model "$LOCAL_MODEL")
  fi
fi

"$PYTHON_BIN" demo.py \
  --dataset "$DATASET" \
  --output "$OUTPUT" \
  --api-base "$LOCAL_BASE" \
  --api-key-env "$LOCAL_KEY_ENV" \
  --screenwriter-model "$LOCAL_MODEL" \
  --visual-editor-model "$LOCAL_MODEL" \
  --caption-editor-model "$LOCAL_MODEL" \
  --director-model "$LOCAL_MODEL" \
  --api-timeout-seconds "${MONTAGE_API_TIMEOUT_SECONDS:-600}" \
  --api-max-retries "${MONTAGE_API_MAX_RETRIES:-2}" \
  "${EXTRA_ARGS[@]}"
