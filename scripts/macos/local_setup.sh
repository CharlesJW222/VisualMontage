#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL_ID="${MONTAGE_LOCAL_MODEL_ID:-mlx-community/Qwen3-VL-8B-Instruct-4bit}"
MODEL_DIR="${MONTAGE_LOCAL_MODEL_DIR:-$PROJECT_ROOT/models/qwen3-vl-8b-instruct-4bit}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "The MLX local VLM backend requires an Apple Silicon macOS machine." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python virtual environment not found: $PYTHON_BIN" >&2
  echo "Run scripts/macos/setup.sh first." >&2
  exit 2
fi

"$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements-macos-local.txt"

"$PYTHON_BIN" - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

model_id, model_dir = sys.argv[1:]
target = Path(model_dir).expanduser().resolve()
target.parent.mkdir(parents=True, exist_ok=True)
print("Downloading Hugging Face model %s" % model_id)
print("Local model directory: %s" % target)
snapshot_download(repo_id=model_id, local_dir=str(target))
print("Model download complete: %s" % target)
PY

echo
echo "Local VLM is ready. Start it in a separate Terminal with:"
echo "bash scripts/macos/local_server.sh"
