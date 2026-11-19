#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_NAME="visualmontage"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found. Install Miniconda with Python 3.8 support, then rerun."
  exit 2
fi

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f "$PROJECT_ROOT/environment.yml" --prune
else
  conda env create -f "$PROJECT_ROOT/environment.yml"
fi

conda run -n "$ENV_NAME" python "$PROJECT_ROOT/scripts/ubuntu/check.py"
echo "Environment ready. Activate with: conda activate $ENV_NAME"
