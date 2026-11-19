#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_index(path: Path) -> List[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot load dataset index %s: %s" % (path, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("Dataset index must be a JSON object")
    datasets = value.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Dataset index requires a non-empty datasets array")
    data_root = value.get("data_root", ".")
    if not isinstance(data_root, str) or not data_root.strip():
        raise ValueError("Dataset index data_root must be a non-empty path")
    dataset_root = (path.parent / data_root).resolve()
    result = []
    seen = set()
    for entry in datasets:
        if not isinstance(entry, dict) or not entry.get("directory"):
            raise ValueError("Each dataset index entry requires directory")
        name = entry["directory"]
        if not isinstance(name, str) or name in (".", "..") or Path(name).name != name:
            raise ValueError("Dataset directory must be a plain directory name")
        if name in seen:
            raise ValueError("Duplicate dataset directory: %s" % name)
        seen.add(name)
        result.append({**entry, "dataset_path": str(dataset_root / name)})
    return result


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


def write_status(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run indexed datasets sequentially via demo.py. Put any demo.py "
            "arguments after --."
        )
    )
    value.add_argument(
        "--output-root",
        default="attack_plan/all_datasets",
        help="Directory containing one output subdirectory per dataset",
    )
    value.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated subset of dataset directory names",
    )
    value.add_argument(
        "--with-validation",
        action="store_true",
        help="Append --enable-victim-validation for every dataset",
    )
    value.add_argument(
        "--resume",
        action="store_true",
        help="Skip datasets whose required output files already exist",
    )
    value.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately instead of continuing after a failed dataset",
    )
    value.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without invoking demo.py",
    )
    return value


def main() -> int:
    args, demo_args = parser().parse_known_args()
    if demo_args and demo_args[0] == "--":
        demo_args = demo_args[1:]
    if "--dataset" in demo_args or "--output" in demo_args:
        raise ValueError(
            "Do not pass --dataset or --output after --; this runner sets them per dataset"
        )
    root = Path(__file__).resolve().parents[1]
    entries = selected_entries(
        load_index(root / "Video Datasets" / "index.json"), args.datasets
    )
    output_root = (root / args.output_root).resolve()
    status_path = output_root / "run_status.jsonl"
    failures = 0

    for entry in entries:
        name = str(entry["directory"])
        dataset = Path(entry["dataset_path"])
        output = output_root / name
        plan_path = output / "montage_plan.json"
        validation_path = output / "victim_validation.json"
        complete = plan_path.is_file() and (
            not args.with_validation or validation_path.is_file()
        )
        if args.resume and complete:
            record = {
                "timestamp": utc_now(),
                "dataset": name,
                "status": "skipped_existing_output",
                "output": str(output),
            }
            write_status(status_path, record)
            print("[skip] %s" % name)
            continue

        command = [
            sys.executable,
            "demo.py",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
        ]
        if args.with_validation:
            command.extend(
                [
                    "--enable-victim-validation",
                    "--target-ground-truth",
                    str(entry.get("ground_truth", "unverified")).lower(),
                ]
            )
        command.extend(demo_args)
        print("[run] %s\n      %s" % (name, " ".join(command)))
        if args.dry_run:
            write_status(
                status_path,
                {
                    "timestamp": utc_now(),
                    "dataset": name,
                    "status": "dry_run",
                    "output": str(output),
                    "command": command,
                },
            )
            continue

        started = utc_now()
        completed = subprocess.run(command, cwd=str(root), check=False)
        record = {
            "started_at": started,
            "finished_at": utc_now(),
            "dataset": name,
            "status": "complete" if completed.returncode == 0 else "failed",
            "exit_code": completed.returncode,
            "output": str(output),
            "command": command,
        }
        write_status(status_path, record)
        if completed.returncode:
            failures += 1
            print("[failed] %s (exit %d)" % (name, completed.returncode))
            if args.stop_on_error:
                break
        else:
            print("[complete] %s" % name)

    print(
        "Finished %d dataset(s); failures=%d; status=%s"
        % (len(entries), failures, status_path)
    )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
