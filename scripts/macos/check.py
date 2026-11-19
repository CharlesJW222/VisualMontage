#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from pathlib import Path
from urllib.parse import urlparse


ROLES = ("screenwriter", "visual_editor", "caption_editor", "director")
VALIDATION_ROLES = ("victim", "judge")
PLACEHOLDER_MARKERS = (
    "replace-with",
    "your-private-api",
    "example.com",
    "your-api",
    "your-key",
    "<api",
)


def is_configured(value):
    text = str(value or "").strip().lower()
    return bool(text) and not any(marker in text for marker in PLACEHOLDER_MARKERS)


def openai_api_base(value):
    try:
        return (urlparse(str(value or "")).hostname or "").lower() == "api.openai.com"
    except ValueError:
        return False


def api_key_status(api_base, configured_key_env):
    if configured_key_env and os.getenv(configured_key_env):
        return configured_key_env, True, "configured environment variable"
    if openai_api_base(api_base) and os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY", True, "OpenAI fallback"
    return configured_key_env, False, "not set"


def _import_check(module_name):
    try:
        module = importlib.import_module(module_name)
        return {
            "available": True,
            "version": str(getattr(module, "__version__", "unknown")),
        }
    except ImportError as exc:
        return {"available": False, "error": str(exc)}


def _role_status(settings, role):
    role_config = ((settings.get("llm") or {}).get("roles") or {}).get(role) or {}
    value = os.getenv("MONTAGE_%s_MODEL" % role.upper()) or role_config.get("model")
    return {
        "configured": is_configured(value),
        "source": "environment" if os.getenv("MONTAGE_%s_MODEL" % role.upper()) else "settings",
    }


def main():
    root = Path(__file__).resolve().parents[2]
    checks = {
        "platform": platform.platform(),
        "macos": platform.system() == "Darwin",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_supported": sys.version_info >= (3, 8),
        "requests": _import_check("requests"),
        "pillow": _import_check("PIL"),
        "configuration": {},
    }

    try:
        settings = json.loads(
            (root / "configs" / "settings.json").read_text(encoding="utf-8")
        )
        agents = json.loads(
            (root / "configs" / "agents.json").read_text(encoding="utf-8")
        )
        llm = settings.get("llm") or {}
        private = llm.get("private_api") or {}
        role_configs = llm.get("roles") or {}
        production_prompts = agents.get("production_team") or {}
        validation_prompts = agents.get("validation_team") or {}
        key_env = os.getenv("MONTAGE_API_KEY_ENV") or private.get(
            "api_key_env", "PRIVATE_API_KEY"
        )
        api_base = os.getenv("MONTAGE_API_BASE") or private.get("base_url")
        active_key_env, key_available, key_source = api_key_status(api_base, key_env)
        checks["configuration"] = {
            "api_base_configured": is_configured(api_base),
            "api_base_source": "environment" if os.getenv("MONTAGE_API_BASE") else "settings",
            "api_key_env": active_key_env,
            "api_key_available": key_available,
            "api_key_source": key_source,
            "models": {role: _role_status(settings, role)["configured"] for role in ROLES},
            "model_entries_present": {
                role: bool((role_configs.get(role) or {}).get("model")) for role in ROLES
            },
            "prompts": {
                role: bool((production_prompts.get(role) or {}).get("system_prompt"))
                for role in ROLES
            },
            "validation_models": {
                role: _role_status(settings, role)["configured"]
                for role in VALIDATION_ROLES
            },
            "validation_model_entries_present": {
                role: bool((role_configs.get(role) or {}).get("model"))
                for role in VALIDATION_ROLES
            },
            "validation_prompts": {
                role: bool((validation_prompts.get(role) or {}).get("system_prompt"))
                for role in VALIDATION_ROLES
            },
        }
    except (OSError, ValueError) as exc:
        checks["configuration"] = {"error": str(exc)}

    config = checks["configuration"]
    local_ready = bool(
        checks["python_supported"]
        and checks["requests"].get("available")
        and checks["pillow"].get("available")
        and config.get("model_entries_present")
        and all(config["model_entries_present"].values())
        and config.get("prompts")
        and all(config["prompts"].values())
    )
    checks["ready_for_local_validation"] = local_ready
    checks["ready_for_api_run"] = bool(
        local_ready
        and config.get("api_base_configured")
        and config.get("api_key_available")
        and config.get("models")
        and all(config["models"].values())
    )
    checks["victim_validation_configured"] = bool(
        local_ready
        and config.get("validation_models")
        and all(config["validation_models"].values())
        and config.get("validation_model_entries_present")
        and all(config["validation_model_entries_present"].values())
        and config.get("validation_prompts")
        and all(config["validation_prompts"].values())
    )
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    return 0 if local_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
