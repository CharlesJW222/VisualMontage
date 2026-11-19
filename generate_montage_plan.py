#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.production import MontagePlan
from src.utils.data_loader import MontageDataset
from src.utils.provenance import write_attack_plan_provenance
from tools import build_production_team


def generate_montage_plan(
    dataset: MontageDataset,
    settings: Dict[str, Any],
    agents_config: Dict[str, Any],
    output_dir: str,
    model_overrides: Optional[Dict[str, str]] = None,
    api_base: Optional[str] = None,
    api_key_env: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    max_retries: Optional[int] = None,
    progress_callback=None,
    verbose: bool = True,
) -> MontagePlan:
    cache_path = str(
        Path(output_dir).expanduser().resolve()
        / ".agent_cache"
        / "prompt_agent_cache.json"
    )
    director = build_production_team(
        settings=settings,
        agents_config=agents_config,
        production_settings=dataset.settings,
        cache_path=cache_path,
        model_overrides=model_overrides,
        api_base=api_base,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        progress_callback=progress_callback,
    )
    return director.manage_debate(dataset, verbose=verbose)


def save_montage_plan(plan: MontagePlan, output_dir: str) -> Path:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    result = plan.to_dict()
    filepath = output_path / "montage_plan.json"
    filepath.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if plan.runtime_diagnostics:
        (output_path / "api_run_stats.json").write_text(
            json.dumps(
                plan.runtime_diagnostics, indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )
    write_attack_plan_provenance(plan, str(output_path))
    if output_path:
        print("Saved: %s" % filepath)
    return filepath
