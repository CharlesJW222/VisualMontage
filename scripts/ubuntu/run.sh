#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

python demo.py --dataset "${MONTAGE_DATASET:-Video_Dataset_Categories/01_Emergency_and_Disaster_Response/air_ground_firefighting}" --output "${MONTAGE_OUTPUT:-attack_plan/air_ground_firefighting}" --screenwriter-model "${MONTAGE_SCREENWRITER_MODEL:-gpt-5.6-luna}" --visual-editor-model "${MONTAGE_VISUAL_EDITOR_MODEL:-gpt-5.6-luna}" --caption-editor-model "${MONTAGE_CAPTION_EDITOR_MODEL:-gpt-5.6-luna}" --director-model "${MONTAGE_DIRECTOR_MODEL:-gpt-5.6-luna}" --api-timeout-seconds "${MONTAGE_API_TIMEOUT_SECONDS:-300}" --api-max-retries "${MONTAGE_API_MAX_RETRIES:-4}" "$@"
