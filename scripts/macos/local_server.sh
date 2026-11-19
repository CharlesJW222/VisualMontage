#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL_DIR="${MONTAGE_LOCAL_MODEL_DIR:-$PROJECT_ROOT/models/qwen3-vl-8b-instruct-4bit}"
HOST="${MONTAGE_LOCAL_HOST:-127.0.0.1}"
PORT="${MONTAGE_LOCAL_PORT:-8081}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python virtual environment not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Local model is not downloaded: $MODEL_DIR" >&2
  echo "Run scripts/macos/local_setup.sh first." >&2
  exit 2
fi

echo "Starting local VLM server"
echo "Model: $MODEL_DIR"
echo "Endpoint: http://$HOST:$PORT/v1"
echo "Keep this Terminal open while demo.py is running."

"$PYTHON_BIN" -m mlx_vlm.server \
  --model "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --max-num-seqs 1 \
  --log-level INFO
