#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    model_dir = Path(
        os.getenv(
            "MONTAGE_LOCAL_MODEL_DIR",
            str(root / "models" / "qwen3-vl-8b-instruct-4bit"),
        )
    ).expanduser()
    checks = {
        "platform": platform.platform(),
        "apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "mlx": {},
        "mlx_vlm": {},
        "huggingface_hub": {},
        "model_directory": str(model_dir),
        "model_downloaded": (model_dir / "config.json").is_file(),
    }
    for module_name, key in (
        ("mlx", "mlx"),
        ("mlx_vlm", "mlx_vlm"),
        ("huggingface_hub", "huggingface_hub"),
    ):
        try:
            module = importlib.import_module(module_name)
            checks[key] = {
                "available": True,
                "version": str(getattr(module, "__version__", "unknown")),
            }
        except ImportError as exc:
            checks[key] = {"available": False, "error": str(exc)}
    checks["ready"] = bool(
        checks["apple_silicon"]
        and checks["mlx"].get("available")
        and checks["mlx_vlm"].get("available")
        and checks["huggingface_hub"].get("available")
        and checks["model_downloaded"]
    )
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    return 0 if checks["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
