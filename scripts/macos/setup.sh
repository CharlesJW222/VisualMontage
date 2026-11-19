#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script is for macOS. Use scripts/ubuntu/setup.sh on Ubuntu." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    CANDIDATE="$(command -v python3)"
    if [[ "$CANDIDATE" != "/usr/bin/python3" ]]; then
      PYTHON_BIN="$CANDIDATE"
    fi
  fi
  if [[ -z "$PYTHON_BIN" ]] && command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.12)"
  fi
  if [[ -z "$PYTHON_BIN" && -x /opt/homebrew/bin/python3 ]]; then
    PYTHON_BIN="/opt/homebrew/bin/python3"
  fi
  if [[ -z "$PYTHON_BIN" && -x /opt/homebrew/bin/python3.12 ]]; then
    PYTHON_BIN="/opt/homebrew/bin/python3.12"
  fi
  if [[ -z "$PYTHON_BIN" && -x /usr/local/bin/python3 ]]; then
    PYTHON_BIN="/usr/local/bin/python3"
  fi
  if [[ -z "$PYTHON_BIN" && -x /usr/local/bin/python3.12 ]]; then
    PYTHON_BIN="/usr/local/bin/python3.12"
  fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3 was not found. Install it with: brew install python@3.12" >&2
  exit 2
fi

if [[ "$PYTHON_BIN" == "/usr/bin/python3" ]]; then
  echo "macOS system python3 is unavailable or managed by Xcode Command Line Tools." >&2
  echo "Install a user Python first: brew install python@3.12" >&2
  echo "Then rerun this script, or set PYTHON_BIN to your Python executable." >&2
  exit 2
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 8):
    raise SystemExit("Python 3.8 or newer is required")
print("Using Python %s" % sys.executable)
PY

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt"
"$VENV_DIR/bin/python" "$PROJECT_ROOT/scripts/macos/check.py"

echo
echo "Environment ready. Activate it with:"
echo "source $VENV_DIR/bin/activate"
echo "Then set MONTAGE_API_BASE and an API key before running demo.py (use OPENAI_API_KEY for api.openai.com)."
