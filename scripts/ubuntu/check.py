#!/usr/bin/env python3
import importlib
import json
import os
import platform
import sys
from pathlib import Path
from urllib.parse import urlparse


ROLES = ("screenwriter", "visual_editor", "caption_editor", "director")
VALIDATION_ROLES = ("victim", "judge")


def is_configured(value):
    text = str(value or "").strip().lower()
    return bool(
        text
        and "replace-with" not in text
        and "your-private-api" not in text
    )


def openai_api_base(value):
    try:
        return (urlparse(str(value or "")).hostname or "").lower() == "api.openai.com"
    except ValueError:
        return False


def api_key_status(api_base, configured_key_env):
    if configured_key_env and os.getenv(configured_key_env):
        return configured_key_env, True
    if openai_api_base(api_base) and os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY", True
    return configured_key_env, False


def main():
    root = Path(__file__).resolve().parents[2]
    checks = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_supported": sys.version_info >= (3, 8),
        "requests": {},
        "pillow": {},
        "configuration": {},
    }
    try:
        module = importlib.import_module("requests")
        checks["requests"] = {
            "available": True,
            "version": str(getattr(module, "__version__", "unknown")),
        }
    except ImportError as exc:
        checks["requests"] = {"available": False, "error": str(exc)}

    try:
        module = importlib.import_module("PIL")
        checks["pillow"] = {
            "available": True,
            "version": str(getattr(module, "__version__", "unknown")),
        }
    except ImportError as exc:
        checks["pillow"] = {"available": False, "error": str(exc)}

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
        prompts = agents.get("production_team") or {}
        api_base = os.getenv("MONTAGE_API_BASE") or private.get("base_url")
        key_env = os.getenv("MONTAGE_API_KEY_ENV") or private.get(
            "api_key_env", "PRIVATE_API_KEY"
        )
        active_key_env, key_available = api_key_status(api_base, key_env)
        checks["configuration"] = {
            "api_base_configured": is_configured(api_base),
            "api_key_env": active_key_env,
            "api_key_available": key_available,
            "models": {
                role: is_configured(
                    os.getenv("MONTAGE_%s_MODEL" % role.upper())
                    or (role_configs.get(role) or {}).get("model")
                )
                for role in ROLES
            },
            "model_entries_present": {
                role: bool((role_configs.get(role) or {}).get("model"))
                for role in ROLES
            },
            "prompts": {
                role: bool((prompts.get(role) or {}).get("system_prompt"))
                for role in ROLES
            },
            "validation_model_entries_present": {
                role: bool((role_configs.get(role) or {}).get("model"))
                for role in VALIDATION_ROLES
            },
            "validation_prompts": {
                role: bool(
                    (
                        (agents.get("validation_team") or {}).get(role) or {}
                    ).get("system_prompt")
                )
                for role in VALIDATION_ROLES
            },
        }
    except (OSError, ValueError) as exc:
        checks["configuration"] = {"error": str(exc)}

    config = checks["configuration"]
    checks["ready_for_local_validation"] = bool(
        checks["python_supported"]
        and checks["requests"].get("available")
        and checks["pillow"].get("available")
        and config.get("model_entries_present")
        and all(config["model_entries_present"].values())
        and config.get("prompts")
        and all(config["prompts"].values())
    )
    checks["ready_for_api_run"] = bool(
        checks["ready_for_local_validation"]
        and config.get("api_base_configured")
        and config.get("api_key_available")
        and config.get("models")
        and all(config["models"].values())
    )
    checks["victim_validation_configured"] = bool(
        checks["ready_for_local_validation"]
        and config.get("validation_model_entries_present")
        and all(config["validation_model_entries_present"].values())
        and config.get("validation_prompts")
        and all(config["validation_prompts"].values())
    )
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    return 0 if checks["ready_for_local_validation"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
