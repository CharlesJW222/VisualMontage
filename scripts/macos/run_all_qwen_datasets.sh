#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
MODEL="${MONTAGE_LOCAL_MODEL_DIR:-$ROOT/models/qwen3-vl-8b-instruct-4bit}"
API_BASE="${MONTAGE_LOCAL_API_BASE:-http://127.0.0.1:8081/v1}"
KEY_ENV="${MONTAGE_LOCAL_API_KEY_ENV:-MONTAGE_LOCAL_API_KEY}"
RUN_LABEL="${MONTAGE_QWEN_ALL_RUN_LABEL:-20260911}"
OUTPUT_ROOT="${MONTAGE_QWEN_ALL_OUTPUT:-$ROOT/attack_plan/QwenAllDatasets_$RUN_LABEL}"
INPUT_ROOT="$OUTPUT_ROOT/input_cache"
STATUS_FILE="$OUTPUT_ROOT/status.jsonl"
RUN_LOG="$OUTPUT_ROOT/runner.log"
MAX_IMAGE_SIDE="${MONTAGE_QWEN_MAX_IMAGE_SIDE:-512}"
IMAGE_QUALITY="${MONTAGE_QWEN_IMAGE_QUALITY:-82}"

export MONTAGE_LOCAL_API_KEY="${MONTAGE_LOCAL_API_KEY:-your key here}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Project Python not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$MODEL/config.json" ]]; then
  echo "Local Qwen model not found: $MODEL" >&2
  echo "Run bash scripts/macos/local_setup.sh first." >&2
  exit 2
fi
if ! curl -fsS "$API_BASE/models" >/dev/null 2>&1; then
  echo "Local Qwen server unavailable at $API_BASE" >&2
  echo "Start it with bash scripts/macos/local_server.sh first." >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$INPUT_ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" | tee -a "$RUN_LOG"
}

record() {
  "$PYTHON_BIN" - "$STATUS_FILE" "$1" "$2" "$3" "$4" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, index, directory, status, message = sys.argv[1:]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "index": int(index),
        "directory": directory,
        "status": status,
        "message": message,
    }, ensure_ascii=False) + "\n")
PY
}

dataset_complete() {
  "$PYTHON_BIN" - "$ROOT" "$1" "$2" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from demo import _target_output_dirs
from src.utils.data_loader import load_dataset

dataset = load_dataset(sys.argv[2], load_images=False)
output = Path(sys.argv[3])
directories = (
    _target_output_dirs(output, dataset.target_ids)
    if len(dataset.target_ids) > 1 else [output]
)
required = ("montage_plan.json", "sequence/captioned_image_sequence.json", "victim_validation.json")
complete = all(
    (directory / name).is_file() and (directory / name).stat().st_size > 0
    for directory in directories for name in required
)
raise SystemExit(0 if complete else 1)
PY
}

DATASETS=()
while IFS= read -r row; do
  DATASETS+=("$row")
done < <("$PYTHON_BIN" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from scripts.run_all_datasets import load_index

for item in sorted(load_index(root / "Video Datasets/index.json"), key=lambda value: int(value["index"])):
    dataset_path = Path(item["dataset_path"])
    manifest = json.loads((dataset_path / "dataset.json").read_text(encoding="utf-8"))
    input_type = "video" if manifest.get("video_file") else "images"
    print(f'{int(item["index"])}\t{item["directory"]}\t{dataset_path}\t{input_type}')
PY
)

if [[ "${#DATASETS[@]}" -eq 0 ]]; then
  echo "No datasets were found in $ROOT/Video Datasets/index.json" >&2
  exit 2
fi

log "Starting all-dataset local Qwen run with model $MODEL"
log "Datasets: ${#DATASETS[@]}; local input max side: $MAX_IMAGE_SIDE px"
log "Output root: $OUTPUT_ROOT"

for row in "${DATASETS[@]}"; do
  IFS=$'\t' read -r index directory dataset_path input_type <<< "$row"
  input_path="$INPUT_ROOT/$directory"
  output_path="$OUTPUT_ROOT/$directory"

  if dataset_complete "$dataset_path" "$output_path"; then
    log "Skipping complete dataset $index ($directory)"
    record "$index" "$directory" "complete" "existing output"
    continue
  fi

  mkdir -p "$output_path"
  if [[ "$input_type" == "video" ]]; then
    input_path="$dataset_path"
  elif [[ ! -s "$input_path/dataset.json" || ! -d "$input_path/images" ]]; then
    log "Dataset $index ($directory): preparing local input copy"
    if ! "$PYTHON_BIN" "$ROOT/scripts/macos/prepare_local_dataset.py" \
      --source "$dataset_path" \
      --destination "$input_path" \
      --max-side "$MAX_IMAGE_SIDE" \
      --quality "$IMAGE_QUALITY" \
      --max-frames 0 >>"$OUTPUT_ROOT/dataset_${index}.log" 2>&1; then
      record "$index" "$directory" "failed" "local input preparation failed"
      log "Dataset $index ($directory): preparation failed; continuing"
      continue
    fi
  fi

  log "Dataset $index ($directory): running production and five Victims"
  if MONTAGE_DATASET="$input_path" \
    MONTAGE_OUTPUT="$output_path" \
    MONTAGE_LOCAL_API_BASE="$API_BASE" \
    MONTAGE_LOCAL_API_KEY_ENV="$KEY_ENV" \
    MONTAGE_LOCAL_MODEL_NAME="$MODEL" \
    PYTHON_BIN="$PYTHON_BIN" \
    bash "$ROOT/scripts/macos/run_local.sh" \
      --enable-victim-validation \
      --victim-model "$MODEL" \
      --judge-model "$MODEL" \
      --num-victims 5 \
      --num-victim-conclusions 1 \
      --target-ground-truth false \
      --api-timeout-seconds "${MONTAGE_API_TIMEOUT_SECONDS:-900}" \
      --api-max-retries "${MONTAGE_API_MAX_RETRIES:-2}" \
      --quiet >>"$OUTPUT_ROOT/dataset_${index}.log" 2>&1; then
    if dataset_complete "$dataset_path" "$output_path"; then
      record "$index" "$directory" "complete" "production and validation complete"
      log "Dataset $index ($directory): complete"
    else
      record "$index" "$directory" "incomplete" "command returned success but an output is missing"
      log "Dataset $index ($directory): incomplete"
    fi
  else
    record "$index" "$directory" "failed" "see dataset_${index}.log"
    log "Dataset $index ($directory): failed; continuing"
  fi
done

log "All-dataset local Qwen run finished"
