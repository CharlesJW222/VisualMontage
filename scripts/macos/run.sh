#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif command -v python3 >/dev/null 2>&1; then
  CANDIDATE="$(command -v python3)"
  if [[ "$CANDIDATE" != "/usr/bin/python3" ]]; then
    PYTHON_BIN="$CANDIDATE"
  fi
fi

if [[ -z "${PYTHON_BIN:-}" ]] && command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.12)"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "Python 3 was not found. Run scripts/macos/setup.sh first." >&2
  exit 2
fi

DATASET="${MONTAGE_DATASET:-Video_Dataset_Categories/01_Emergency_and_Disaster_Response/air_ground_firefighting}"
OUTPUT="${MONTAGE_OUTPUT:-attack_plan/x59_simulator_to_first_flight}"
API_TIMEOUT="${MONTAGE_API_TIMEOUT_SECONDS:-300}"
API_RETRIES="${MONTAGE_API_MAX_RETRIES:-4}"

"$PYTHON_BIN" demo.py --dataset "$DATASET" --output "$OUTPUT" --screenwriter-model "${MONTAGE_SCREENWRITER_MODEL:-gpt-5.6-luna}" --visual-editor-model "${MONTAGE_VISUAL_EDITOR_MODEL:-gpt-5.6-luna}" --caption-editor-model "${MONTAGE_CAPTION_EDITOR_MODEL:-gpt-5.6-luna}" --director-model "${MONTAGE_DIRECTOR_MODEL:-gpt-5.6-luna}" --api-timeout-seconds "$API_TIMEOUT" --api-max-retries "$API_RETRIES" "$@"
