#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from PIL import Image


DEFAULT_OUTPUT_ROOT = "attack_plan/Good Results"
DEFAULT_AGGREGATE_NAME = "good_results_all_models_validation.json"
DEFAULT_MAX_IMAGE_BYTES = 30_000_000
DEFAULT_COMPRESSED_MAX_SIDE = 1024
DEFAULT_COMPRESSED_QUALITY = 82


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_index(path: Path) -> List[Dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    entries = value.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Dataset index requires a non-empty datasets array")
    return [dict(entry) for entry in entries]


def validate_filename(name: str) -> str:
    candidate = Path(name)
    if candidate.name != name or candidate.suffix != ".json":
        raise ValueError("--aggregate-name must be a plain .json filename")
    return name


def discover_datasets(
    output_root: Path, index_entries: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    indexed = {
        str(entry["directory"]).casefold(): dict(entry)
        for entry in index_entries
        if entry.get("directory")
    }
    found: Dict[str, Dict[str, Any]] = {}
    for directory in output_root.iterdir():
        if not directory.is_dir():
            continue
        sequence = directory / "sequence" / "captioned_image_sequence.json"
        entry = indexed.get(directory.name.casefold())
        if entry and sequence.is_file():
            entry["_directory_on_disk"] = directory.name
            found[str(entry["directory"]).casefold()] = entry
    return [
        found[key]
        for key in sorted(
            found,
            key=lambda item: int(found[item].get("index", 10**9)),
        )
    ]


def discover_models(output_root: Path) -> List[str]:
    models: Set[str] = set()
    for path in output_root.glob("*/victim_validation*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for model in (value.get("metadata") or {}).get("victim_models") or []:
            if model:
                models.add(str(model))
    return sorted(models)


def compact_result(value: Dict[str, Any]) -> Dict[str, Any]:
    analysis = dict(value.get("analysis") or {})
    adjudication = dict(value.get("adjudication") or {})
    compact_adjudication: Dict[str, Any] = {}
    for key in ("majority_vote", "ai_judge", "metrics"):
        item = adjudication.get(key)
        if isinstance(item, dict):
            item = dict(item)
            item.pop("raw_response", None)
            compact_adjudication[key] = item
    victims = []
    for item in value.get("victim_verdicts") or []:
        if not isinstance(item, dict):
            continue
        victims.append(
            {
                key: item.get(key)
                for key in (
                    "victim_id",
                    "model",
                    "verdict",
                    "confidence",
                    "reasoning",
                )
            }
        )
    return {
        "metadata": value.get("metadata") or {},
        "analysis": analysis,
        "adjudication": compact_adjudication,
        "victim_verdicts": victims,
    }


def resolve_sequence_image(sequence_path: Path, value: Any) -> Path:
    image_path = Path(str(value)).expanduser()
    if image_path.is_file():
        return image_path.resolve()
    fallback = sequence_path.parent / "victim_images" / image_path.name
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError("Visible image not found: %s" % image_path)


def prepare_sequence_for_request(
    sequence_path: Path,
    destination: Path,
    max_image_bytes: int,
    max_side: int,
    quality: int,
) -> Tuple[Path, bool]:
    raw = json.loads(sequence_path.read_text(encoding="utf-8"))
    items = raw.get("items") or []
    image_paths = [
        resolve_sequence_image(sequence_path, item["image_path"])
        for item in items
        if isinstance(item, dict)
        and not item.get("image_withheld", False)
        and item.get("image_path")
    ]
    total_bytes = sum(path.stat().st_size for path in image_paths)
    if total_bytes <= max_image_bytes:
        return sequence_path, False

    image_root = destination / "visible_images"
    image_root.mkdir(parents=True, exist_ok=True)
    copied = {}
    for position, item in enumerate(items, start=1):
        if (
            not isinstance(item, dict)
            or item.get("image_withheld", False)
            or not item.get("image_path")
        ):
            continue
        source = resolve_sequence_image(sequence_path, item["image_path"])
        output = image_root / ("item_%03d.jpg" % position)
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            image.save(output, format="JPEG", quality=quality, optimize=True)
        copied[position] = str(output.resolve())

    rewritten = json.loads(json.dumps(raw))
    for position, item in enumerate(rewritten.get("items") or [], start=1):
        if position in copied:
            item["image_path"] = copied[position]
    rewritten["validation_input_copy"] = {
        "source_sequence": str(sequence_path.resolve()),
        "source_image_bytes": total_bytes,
        "max_image_side": max_side,
        "jpeg_quality": quality,
        "reason": "automatic_request_size_guard",
    }
    output_sequence = destination / "captioned_image_sequence.json"
    output_sequence.write_text(
        json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_sequence, True


def empty_state(
    output_root: Path,
    datasets: List[Dict[str, Any]],
    models: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "schema_version": "visualmontage.aggregate_validation.v1",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "output_root": str(output_root),
        "datasets": [str(entry["directory"]) for entry in datasets],
        "models": models,
        "configuration": {
            "num_victims": args.num_victims,
            "num_conclusions": args.num_conclusions,
            "high_confidence_threshold": args.high_confidence_threshold,
            "api_base": args.api_base,
            "api_key_env": args.api_key_env,
        },
        "runs": [],
        "summary": {},
    }


def write_state(path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def summarize(state: Dict[str, Any]) -> Dict[str, Any]:
    model_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    dataset_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run in state.get("runs") or []:
        if run.get("status") != "complete":
            continue
        result = run.get("result") or {}
        analysis = result.get("analysis") or {}
        adjudication = result.get("adjudication") or {}
        judge = adjudication.get("ai_judge") or {}
        row = {
            "model": run.get("model"),
            "dataset": run.get("dataset"),
            "valid_victims": int(analysis.get("valid_victims") or 0),
            "victims_deceived": int(analysis.get("victims_deceived") or 0),
            "ASR": float(analysis.get("ASR") or 0),
            "HC_ASR": float(analysis.get("HC_ASR") or 0),
            "judge_decision": judge.get("decision"),
            "judge_confidence": judge.get("confidence"),
        }
        model_groups[str(run.get("model"))].append(row)
        dataset_groups[str(run.get("dataset"))].append(row)

    model_summary = {}
    for model, rows in sorted(model_groups.items()):
        valid = sum(row["valid_victims"] for row in rows)
        fooled = sum(row["victims_deceived"] for row in rows)
        judge_count = sum(row["judge_decision"] in ("accept", "reject") for row in rows)
        accepts = sum(row["judge_decision"] == "accept" for row in rows)
        model_summary[model] = {
            "datasets_complete": len(rows),
            "valid_victims": valid,
            "victims_deceived": fooled,
            "ASR": round(100 * fooled / valid, 2) if valid else None,
            "judge_accepts": accepts,
            "judge_evaluated": judge_count,
            "judge_accept_rate": round(100 * accepts / judge_count, 2)
            if judge_count
            else None,
        }

    dataset_summary = {}
    for dataset, rows in sorted(dataset_groups.items()):
        dataset_summary[dataset] = {
            "models": {
                row["model"]: {
                    "ASR": row["ASR"],
                    "victims_deceived": row["victims_deceived"],
                    "valid_victims": row["valid_victims"],
                    "judge_decision": row["judge_decision"],
                    "judge_confidence": row["judge_confidence"],
                }
                for row in rows
            }
        }
    return {"by_model": model_summary, "by_dataset": dataset_summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate existing Good Results sequences with all previously "
            "observed models and write one aggregate JSON document."
        )
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated Good Results directory names to validate",
    )
    parser.add_argument("--models", default=None)
    parser.add_argument("--aggregate-name", default=DEFAULT_AGGREGATE_NAME)
    parser.add_argument("--num-victims", type=int, default=5)
    parser.add_argument("--num-conclusions", type=int, default=1)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-timeout-seconds", type=int, default=600)
    parser.add_argument("--api-max-retries", type=int, default=4)
    parser.add_argument("--max-image-bytes", type=int, default=DEFAULT_MAX_IMAGE_BYTES)
    parser.add_argument("--compressed-max-side", type=int, default=DEFAULT_COMPRESSED_MAX_SIDE)
    parser.add_argument("--compressed-quality", type=int, default=DEFAULT_COMPRESSED_QUALITY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def run(args: argparse.Namespace, project_root: Path) -> int:
    if args.num_victims < 1 or args.num_conclusions < 1:
        raise ValueError("--num-victims and --num-conclusions must be >= 1")
    output_root = (project_root / args.output_root).resolve()
    aggregate_path = output_root / validate_filename(args.aggregate_name)
    entries = discover_datasets(
        output_root, load_index(project_root / "Video Datasets" / "index.json")
    )
    if not entries:
        raise ValueError("No Good Results sequences were found")
    if args.datasets:
        requested = [
            item.strip().casefold()
            for item in args.datasets.split(",")
            if item.strip()
        ]
        available = {
            str(entry["directory"]).casefold(): entry for entry in entries
        }
        missing = [item for item in requested if item not in available]
        if missing:
            raise ValueError(
                "Unknown Good Results dataset(s): %s"
                % ", ".join(missing)
            )
        entries = [available[item] for item in requested]
    models = (
        [item.strip() for item in args.models.split(",") if item.strip()]
        if args.models
        else discover_models(output_root)
    )
    if not models:
        raise ValueError("No validation models were found")

    if args.resume and aggregate_path.is_file():
        state = json.loads(aggregate_path.read_text(encoding="utf-8"))
    else:
        state = empty_state(output_root, entries, models, args)
    completed = {
        (run.get("model"), run.get("dataset"))
        for run in state.get("runs") or []
        if run.get("status") == "complete"
    }
    failures = 0
    with tempfile.TemporaryDirectory(prefix="good_results_validation_") as temp_dir:
        temp_root = Path(temp_dir)
        for model in models:
            for entry in entries:
                dataset = str(entry["directory"])
                key = (model, dataset)
                if args.resume and key in completed:
                    print("[skip] %s / %s" % (model, dataset))
                    continue
                sequence = output_root / str(entry.get("_directory_on_disk", dataset)) / "sequence" / "captioned_image_sequence.json"
                temp_output = temp_root / ("%s__%s.json" % (model.replace("/", "_"), dataset))
                prepared_sequence, compressed = prepare_sequence_for_request(
                    sequence,
                    temp_root / ("images__%s__%s" % (model.replace("/", "_"), dataset)),
                    args.max_image_bytes,
                    args.compressed_max_side,
                    args.compressed_quality,
                )
                command = [
                    sys.executable,
                    "victim_validation.py",
                    "--sequence",
                    str(prepared_sequence),
                    "--output",
                    str(temp_output),
                    "--target-ground-truth",
                    str(entry.get("ground_truth", "unverified")).lower(),
                    "--num-victims",
                    str(args.num_victims),
                    "--num-conclusions",
                    str(args.num_conclusions),
                    "--high-confidence-threshold",
                    str(args.high_confidence_threshold),
                    "--victim-model",
                    model,
                    "--judge-model",
                    model,
                    "--api-base",
                    args.api_base,
                    "--api-key-env",
                    args.api_key_env,
                    "--api-timeout-seconds",
                    str(args.api_timeout_seconds),
                    "--api-max-retries",
                    str(args.api_max_retries),
                    "--quiet",
                ]
                print("[run] %s / %s" % (model, dataset), flush=True)
                completed_process = subprocess.run(
                    command,
                    cwd=str(project_root),
                    check=False,
                    capture_output=True,
                    text=True,
                )
                run_record: Dict[str, Any] = {
                    "model": model,
                    "dataset": dataset,
                    "started_at": utc_now(),
                    "exit_code": completed_process.returncode,
                    "status": "failed",
                    "input_images_compressed": compressed,
                }
                if completed_process.returncode == 0 and temp_output.is_file():
                    value = json.loads(temp_output.read_text(encoding="utf-8"))
                    run_record["status"] = "complete"
                    run_record["result"] = compact_result(value)
                    print("[complete] %s / %s" % (model, dataset), flush=True)
                else:
                    failures += 1
                    run_record["error_tail"] = (
                        (completed_process.stderr or completed_process.stdout or "")[-2000:]
                    )
                    print("[failed] %s / %s" % (model, dataset), flush=True)
                state.setdefault("runs", []).append(run_record)
                state["summary"] = summarize(state)
                write_state(aggregate_path, state)
                if failures and args.stop_on_error:
                    break
            if failures and args.stop_on_error:
                break
    state["summary"] = summarize(state)
    write_state(aggregate_path, state)
    print("Aggregate result: %s" % aggregate_path)
    print("Models: %d; datasets: %d; failures: %d" % (len(models), len(entries), failures))
    return 1 if failures else 0


def main() -> int:
    args = build_parser().parse_args()
    return run(args, Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
