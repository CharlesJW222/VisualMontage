#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_OUTPUT_ROOT = "attack_plan/all_datasets_5.6luna_gen_5.5vali"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_RESULT_NAME = "victim_validation_gpt54.json"
DEFAULT_STATUS_NAME = "revalidation_gpt54_status.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_index(path: Path) -> List[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot load dataset index %s: %s" % (path, exc)) from exc
    datasets = value.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Dataset index requires a non-empty datasets array")
    entries = []
    for entry in datasets:
        if not isinstance(entry, dict) or not entry.get("directory"):
            raise ValueError("Each dataset index entry requires directory")
        entries.append(dict(entry))
    return entries


def selected_entries(
    entries: Iterable[Dict[str, Any]], requested: Optional[str]
) -> List[Dict[str, Any]]:
    if not requested:
        return list(entries)
    requested_names = {
        value.strip() for value in requested.split(",") if value.strip()
    }
    selected = [entry for entry in entries if entry["directory"] in requested_names]
    missing = requested_names - {entry["directory"] for entry in selected}
    if missing:
        raise ValueError(
            "Datasets not present in Video Datasets/index.json: %s"
            % ", ".join(sorted(missing))
        )
    return selected


def validate_plain_filename(name: str, suffix: str, option: str) -> str:
    candidate = Path(name)
    if candidate.name != name or candidate.suffix != suffix:
        raise ValueError("%s must be a plain %s filename" % (option, suffix))
    return name


def validate_result_name(name: str) -> str:
    return validate_plain_filename(name, ".json", "--result-name")


def validate_status_name(name: str) -> str:
    return validate_plain_filename(name, ".jsonl", "--status-name")


def write_status(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def result_summary(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    analysis = value.get("analysis")
    adjudication = value.get("adjudication")
    if not isinstance(analysis, dict) or not isinstance(adjudication, dict):
        return {}
    judge = adjudication.get("ai_judge") or adjudication.get("judge_agent")
    summary: Dict[str, Any] = {"metrics": analysis}
    if isinstance(judge, dict):
        summary["ai_judge"] = {
            "decision": judge.get("decision"),
            "confidence": judge.get("confidence"),
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fresh Victim/Judge validation over existing captioned "
            "sequences without rerunning the production pipeline."
        )
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated subset of dataset directory names",
    )
    parser.add_argument("--victim-model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--num-victims", type=int, default=5)
    parser.add_argument("--num-conclusions", type=int, default=1)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--result-name", default=DEFAULT_RESULT_NAME)
    parser.add_argument("--status-name", default=DEFAULT_STATUS_NAME)
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-timeout-seconds", type=int, default=600)
    parser.add_argument("--api-max-retries", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_revalidation(args: argparse.Namespace, root: Path) -> int:
    result_name = validate_result_name(args.result_name)
    status_name = validate_status_name(args.status_name)
    if args.num_victims < 1:
        raise ValueError("--num-victims must be >= 1")
    if args.num_conclusions < 1:
        raise ValueError("--num-conclusions must be >= 1")

    entries = selected_entries(
        load_index(root / "Video Datasets" / "index.json"), args.datasets
    )
    output_root = (root / args.output_root).resolve()
    status_path = output_root / status_name
    failures = 0

    for entry in entries:
        name = str(entry["directory"])
        dataset_output = output_root / name
        sequence = dataset_output / "sequence" / "captioned_image_sequence.json"
        result_path = dataset_output / result_name
        record_base = {
            "dataset": name,
            "sequence": str(sequence),
            "output": str(result_path),
            "victim_model": args.victim_model,
            "judge_model": args.judge_model,
        }

        if not sequence.is_file():
            write_status(
                status_path,
                {
                    **record_base,
                    "timestamp": utc_now(),
                    "status": "skipped_missing_sequence",
                },
            )
            print("[skip] %s (captioned sequence is missing)" % name)
            continue
        if args.resume and result_path.is_file() and result_path.stat().st_size > 0:
            write_status(
                status_path,
                {
                    **record_base,
                    "timestamp": utc_now(),
                    "status": "skipped_existing_result",
                    **result_summary(result_path),
                },
            )
            print("[skip] %s" % name)
            continue

        command = [
            sys.executable,
            "victim_validation.py",
            "--sequence",
            str(sequence),
            "--output",
            str(result_path),
            "--target-ground-truth",
            str(entry.get("ground_truth", "unverified")).lower(),
            "--num-victims",
            str(args.num_victims),
            "--num-conclusions",
            str(args.num_conclusions),
            "--high-confidence-threshold",
            str(args.high_confidence_threshold),
            "--victim-model",
            args.victim_model,
            "--judge-model",
            args.judge_model,
            "--api-base",
            args.api_base,
            "--api-key-env",
            args.api_key_env,
            "--api-timeout-seconds",
            str(args.api_timeout_seconds),
            "--api-max-retries",
            str(args.api_max_retries),
        ]
        print("[run] %s\n      %s" % (name, " ".join(command)))
        if args.dry_run:
            write_status(
                status_path,
                {
                    **record_base,
                    "timestamp": utc_now(),
                    "status": "dry_run",
                    "command": command,
                },
            )
            continue

        started = utc_now()
        completed = subprocess.run(command, cwd=str(root), check=False)
        record = {
            **record_base,
            "started_at": started,
            "finished_at": utc_now(),
            "status": "complete" if completed.returncode == 0 else "failed",
            "exit_code": completed.returncode,
            "command": command,
        }
        if completed.returncode == 0:
            record.update(result_summary(result_path))
            print("[complete] %s" % name)
        else:
            failures += 1
            print("[failed] %s (exit %d)" % (name, completed.returncode))
        write_status(status_path, record)
        if completed.returncode and args.stop_on_error:
            break

    print(
        "Finished %d dataset(s); failures=%d; status=%s"
        % (len(entries), failures, status_path)
    )
    return 1 if failures else 0


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    return run_revalidation(args, root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
