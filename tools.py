from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.production import CaptionEditor, Director, Screenwriter, VisualEditor
from src.utils.agent_runtime import AgentCache, PromptAgent
from src.utils.data_loader import MontageError, ProductionSettings
from src.utils.llm_client import create_private_client


ROLE_ENV = {
    "screenwriter": "MONTAGE_SCREENWRITER_MODEL",
    "visual_editor": "MONTAGE_VISUAL_EDITOR_MODEL",
    "caption_editor": "MONTAGE_CAPTION_EDITOR_MODEL",
    "director": "MONTAGE_DIRECTOR_MODEL",
}

def load_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise MontageError("%s not found: %s" % (label, path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MontageError("Invalid %s JSON: %s" % (label, exc)) from exc
    if not isinstance(value, dict):
        raise MontageError("%s must contain a JSON object" % label)
    return value


def load_configs(config_dir: str = "configs") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    root = Path(config_dir).expanduser().resolve()
    settings = load_json(root / "settings.json", "settings config")
    agents = load_json(root / "agents.json", "agents config")
    return settings, agents


def apply_production_overrides(dataset, args) -> ProductionSettings:
    values = dataset.settings.to_dict()
    has_range_override = any(
        getattr(args, name, None) is not None
        for name in ("min_output_frames", "max_output_frames")
    )
    if getattr(args, "output_frames", None) is not None:
        if has_range_override:
            raise MontageError(
                "Use either --output-frames or --min-output-frames/--max-output-frames"
            )
        values.pop("min_output_frames", None)
        values.pop("max_output_frames", None)
    elif has_range_override:
        values.pop("output_frames", None)
    override_names = (
        "screenwriter_rounds",
        "visual_editor_rounds",
        "caption_editor_rounds",
        "output_frames",
        "min_output_frames",
        "max_output_frames",
        "tau",
        "epsilon",
        "search_alpha",
        "search_beta",
    )
    for name in override_names:
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    settings = ProductionSettings.from_dict(values)
    if settings.frame_selection_bounds[0] > len(dataset.image_units):
        raise MontageError(
            "frame selection count exceeds the image-frame count"
        )
    dataset.settings = settings
    return settings


def _role_model(role: str, role_config: Dict[str, Any], override: Optional[str]) -> str:
    model = override or os.getenv(ROLE_ENV[role]) or role_config.get("model")
    if not model:
        raise MontageError("Missing VLM model for %s" % role)
    return str(model)


def _role_prompt(agents_config: Dict[str, Any], role: str) -> str:
    production_team = agents_config.get("production_team") or {}
    value = production_team.get(role) or {}
    prompt = value.get("system_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise MontageError(
            "Missing system_prompt for production_team.%s in agents.json" % role
        )
    return prompt


def _build_client(
    settings: Dict[str, Any],
    role: str,
    model: str,
    api_base: Optional[str],
    api_key_env: Optional[str],
    timeout_seconds: Optional[int],
    max_retries: Optional[int],
):
    llm = settings.get("llm") or {}
    private = dict(llm.get("private_api") or {})
    role_config = dict((llm.get("roles") or {}).get(role) or {})
    values = dict(private)
    values.update(role_config)
    values["model"] = model
    configured_timeout = timeout_seconds
    if configured_timeout is None:
        configured_timeout = os.getenv("MONTAGE_API_TIMEOUT_SECONDS")
    if configured_timeout is None:
        configured_timeout = os.getenv("MONTAGE_API_READ_TIMEOUT_SECONDS")
    if configured_timeout is not None:
        values["timeout_seconds"] = int(configured_timeout)
    configured_retries = max_retries
    if configured_retries is None:
        configured_retries = os.getenv("MONTAGE_API_MAX_RETRIES")
    if configured_retries is not None:
        values["max_retries"] = int(configured_retries)
    configured_connect_timeout = os.getenv("MONTAGE_API_CONNECT_TIMEOUT_SECONDS")
    if configured_connect_timeout is not None:
        values["connect_timeout_seconds"] = int(configured_connect_timeout)
    return create_private_client(
        values,
        {
            "base_url": api_base or os.getenv("MONTAGE_API_BASE") or values.get("base_url"),
            "api_key_env": (
                api_key_env
                or os.getenv("MONTAGE_API_KEY_ENV")
                or values.get("api_key_env")
            ),
            "model": model,
        },
    )


def build_production_team(
    settings: Dict[str, Any],
    agents_config: Dict[str, Any],
    production_settings: ProductionSettings,
    cache_path: str,
    model_overrides: Optional[Dict[str, str]] = None,
    api_base: Optional[str] = None,
    api_key_env: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    max_retries: Optional[int] = None,
    progress_callback=None,
):
    overrides = dict(model_overrides or {})
    role_configs = (settings.get("llm") or {}).get("roles") or {}
    cache = AgentCache(cache_path)
    prompt_agents = {}
    for role in ROLE_ENV:
        role_config = dict(role_configs.get(role) or {})
        model = _role_model(role, role_config, overrides.get(role))
        role_cache = AgentCache(None) if role == "visual_editor" else cache
        prompt_agents[role] = PromptAgent(
            role,
            _build_client(
                settings,
                role,
                model,
                api_base,
                api_key_env,
                timeout_seconds,
                max_retries,
            ),
            _role_prompt(agents_config, role),
            role_cache,
            production_settings.json_retries,
            progress_callback,
        )

    screenwriter = Screenwriter(prompt_agents["screenwriter"])
    visual_editor = VisualEditor(prompt_agents["visual_editor"])
    caption_editor = CaptionEditor(prompt_agents["caption_editor"])
    director = Director(
        prompt_agents["director"],
        screenwriter,
        visual_editor,
        caption_editor,
        progress_callback=progress_callback,
    )
    return director
