#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_dataset, resolve_dataset_file
from tools import load_configs


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_directory_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return name or "target"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one attack plan per target in a dataset manifest"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--target-ids",
        default=None,
        help="Optional comma-separated target ids; default is every target",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--screenwriter-model", default=None)
    parser.add_argument("--visual-editor-model", default=None)
    parser.add_argument("--caption-editor-model", default=None)
    parser.add_argument("--director-model", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--api-timeout-seconds", type=int, default=600)
    parser.add_argument("--api-max-retries", type=int, default=4)
    parser.add_argument("--status-name", default="multi_target_status.jsonl")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    settings, _agents = load_configs(str(project_root / "configs"))
    dataset_path = resolve_dataset_file(args.dataset)
    dataset = load_dataset(str(dataset_path))

    requested = None
    if args.target_ids:
        requested = {
            item.strip()
            for item in args.target_ids.split(",")
            if item.strip()
        }
        missing = requested - set(dataset.target_ids)
        if missing:
            raise ValueError(
                "Unknown target id(s): %s" % ", ".join(sorted(missing))
            )

    default_output_root = Path("attack_plan") / Path(dataset.root).name
    output_root = Path(args.output_root or default_output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / args.status_name

    common = []
    if args.model:
        common.extend(
            [
                "--screenwriter-model",
                args.model,
                "--visual-editor-model",
                args.model,
                "--caption-editor-model",
                args.model,
                "--director-model",
                args.model,
            ]
        )
    else:
        for flag, value in (
            ("--screenwriter-model", args.screenwriter_model),
            ("--visual-editor-model", args.visual_editor_model),
            ("--caption-editor-model", args.caption_editor_model),
            ("--director-model", args.director_model),
        ):
            if value:
                common.extend([flag, value])
    for flag, value in (
        ("--api-base", args.api_base),
        ("--api-key-env", args.api_key_env),
    ):
        if value:
            common.extend([flag, value])
    common.extend(
        [
            "--api-timeout-seconds",
            str(args.api_timeout_seconds),
            "--api-max-retries",
            str(args.api_max_retries),
        ]
    )
    if args.quiet:
        common.append("--quiet")

    failures = 0
    with status_path.open("a", encoding="utf-8") as status:
        for target in dataset.targets or []:
            if requested is not None and target.target_id not in requested:
                continue
            target_output = output_root / safe_directory_name(target.target_id)
            command = [
                sys.executable,
                "demo.py",
                "--dataset",
                str(dataset_path),
                "--target-id",
                target.target_id,
                "--output",
                str(target_output),
                *common,
            ]
            print("[run] %s -> %s" % (target.target_id, target_output), flush=True)
            completed = subprocess.run(command, cwd=str(project_root), check=False)
            record = {
                "timestamp": utc_now(),
                "target_id": target.target_id,
                "target_label": target.label,
                "output": str(target_output),
                "exit_code": completed.returncode,
                "status": "complete" if completed.returncode == 0 else "failed",
                "command": command,
            }
            status.write(json.dumps(record, ensure_ascii=False) + "\n")
            status.flush()
            if completed.returncode != 0:
                failures += 1
                print("[failed] %s" % target.target_id, file=sys.stderr)

    print("Targets complete: %d; failures: %d" % (len(dataset.targets or []) - failures, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
