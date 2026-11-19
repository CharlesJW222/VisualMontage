#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from generate_montage_plan import generate_montage_plan, save_montage_plan
from src.environment import DistributedImagePublisher, ImagePublisher
from src.utils.data_loader import MontageError, load_dataset
from src.utils.provenance import write_attack_plan_provenance
from src.utils.video_sampler import sample_video
from tools import apply_production_overrides, load_configs
from victim_validation import (
    build_validation_client,
    resolve_victim_configuration,
    run_victim_validation,
    save_victim_validation,
    validation_prompt,
)


class ProgressReporter:
    def __init__(self, output_dir: Path, quiet: bool = False):
        self.path = output_dir / "run_progress.json"
        self.quiet = quiet
        self.started_at = datetime.now().isoformat()
        self.lock = threading.Lock()
        self.counts = {}

    def __call__(self, event):
        event = dict(event or {})
        name = str(event.get("event") or "progress")
        with self.lock:
            self.counts[name] = self.counts.get(name, 0) + 1
            snapshot = {
                "schema_version": "visual_montage.prompt_agent_progress.v1",
                "started_at": self.started_at,
                "updated_at": datetime.now().isoformat(),
                "last_event": event,
                "event_counts": dict(self.counts),
            }
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(snapshot, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(self.path)
            if not self.quiet:
                details = ""
                if event.get("agent"):
                    details += " agent=%s" % event["agent"]
                if event.get("model"):
                    details += " model=%s" % event["model"]
                print(
                    "[%s] %s%s"
                    % (datetime.now().strftime("%H:%M:%S"), name, details),
                    file=sys.stderr,
                    flush=True,
                )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Four prompt-driven VLM agents for authentic-image montage"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset directory or dataset.json (default from settings.json)",
    )
    parser.add_argument(
        "--target-id",
        default=None,
        help="Run this target only; otherwise run every user-defined target",
    )
    parser.add_argument("--video", default=None, help="Local video to use as the candidate frame pool")
    parser.add_argument(
        "--frame-interval", type=float, default=3.0,
        help="Deterministic video sampling interval in seconds (default: 3.0)",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--screenwriter-model", default=None)
    parser.add_argument("--visual-editor-model", default=None)
    parser.add_argument("--caption-editor-model", default=None)
    parser.add_argument("--director-model", default=None)
    parser.add_argument("--screenwriter-rounds", type=int, default=None)
    parser.add_argument("--visual-editor-rounds", type=int, default=None)
    parser.add_argument("--caption-editor-rounds", type=int, default=None)
    parser.add_argument(
        "--output-frames",
        type=int,
        default=None,
        help="Legacy fixed frame count (cannot be combined with range overrides)",
    )
    parser.add_argument("--min-output-frames", type=int, default=None)
    parser.add_argument("--max-output-frames", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--search-alpha", type=float, default=None)
    parser.add_argument("--search-beta", type=float, default=None)
    parser.add_argument("--api-timeout-seconds", type=int, default=None)
    parser.add_argument("--api-max-retries", type=int, default=None)
    parser.add_argument("--enable-victim-validation", action="store_true")
    parser.add_argument("--num-victims", type=int, default=None)
    parser.add_argument("--num-victim-conclusions", type=int, default=None)
    victim_model_group = parser.add_mutually_exclusive_group()
    victim_model_group.add_argument("--victim-model", default=None)
    victim_model_group.add_argument(
        "--victim-models",
        default=None,
        help="Comma-separated models; one independent Victim per model",
    )
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--target-ground-truth",
        choices=("true", "false", "unverified"),
        default=None,
    )
    parser.add_argument(
        "--victim-confidence-threshold", type=float, default=None
    )
    parser.add_argument("--skip-ai-judge", action="store_true")
    parser.add_argument("--victim-no-cot", action="store_true")
    parser.add_argument("--no-materialize", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _resolve_dataset(args, settings):
    dataset_config = settings.get("dataset") or {}
    if args.dataset:
        return args.dataset
    data_root = Path(dataset_config.get("data_root", "Video Datasets"))
    default_dataset = dataset_config.get("default_dataset", "A0FZIwabctw")
    return str(data_root / default_dataset)


def _resolve_video(args, dataset_path):
    if args.video:
        return args.video
    manifest_path = Path(dataset_path).expanduser()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "dataset.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    video_file = manifest.get("video_file")
    if not isinstance(video_file, str) or not video_file.strip():
        return None
    video_path = Path(video_file).expanduser()
    if not video_path.is_absolute():
        video_path = manifest_path.parent / video_path
    return str(video_path.resolve())


def _resolve_output_dir(args, settings, dataset):
    default_output = (settings.get("output") or {}).get(
        "directory", "attack_plan"
    )
    dataset_directory = Path(dataset.root).name
    return Path(
        args.output or Path(default_output) / dataset_directory
    ).expanduser().resolve()


def main():
    args = build_parser().parse_args()
    settings, agents_config = load_configs(args.config_dir)
    dataset_path = _resolve_dataset(args, settings)
    args.video = _resolve_video(args, dataset_path)
    dataset = load_dataset(
        dataset_path, load_images=not bool(args.video), validate_frame_count=False,
        settings_defaults=settings.get("production"),
    )
    if args.target_id:
        dataset = dataset.select_target(args.target_id)
    if args.enable_victim_validation and args.no_materialize:
        raise MontageError(
            "Victim Validation requires materialized victim-visible images; "
            "remove --no-materialize"
        )

    output_dir = _resolve_output_dir(args, settings, dataset)
    if args.video:
        dataset.image_units = sample_video(args.video, str(output_dir), args.frame_interval)
    apply_production_overrides(dataset, args)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.target_id or len(dataset.targets or []) == 1:
        runs = [(dataset, output_dir)]
    else:
        directories = _target_output_dirs(output_dir, dataset.target_ids)
        runs = [
            (dataset.select_target(target_id), directory)
            for target_id, directory in zip(dataset.target_ids, directories)
        ]
    for target_dataset, target_output in runs:
        target_output.mkdir(parents=True, exist_ok=True)
        _run_target(args, settings, agents_config, target_dataset, target_output)
    return 0


def _target_output_dirs(output_dir, target_ids):
    used = {"candidate_frames", "sampling_metadata.json"}
    directories = []
    for target_id in target_ids:
        name = quote(target_id, safe="._-")
        if name in (".", ".."):
            name = name.replace(".", "%2E")
        if len(name) > 200 or name.casefold() in used:
            suffix = hashlib.sha256(target_id.encode("utf-8")).hexdigest()[:16]
            name = name[:180] + "__" + suffix
        while name.casefold() in used:
            name += "_"
        used.add(name.casefold())
        directory = output_dir / name
        if directory.is_symlink():
            raise MontageError("Target output must not be a symbolic link: %s" % directory)
        directories.append(directory)
    return directories


def _run_target(args, settings, agents_config, dataset, output_dir):
    verbose = not args.quiet
    if verbose:
        print("\n%s" % ("=" * 80))
        print("GENERATE IMAGE MONTAGE PLAN")
        print("%s" % ("=" * 80))
        print("Dataset       : %s" % dataset.dataset_id)
        print("Target        : %s" % dataset.target.label)
        print("Image frames  : %d" % len(dataset.image_units))
        print(
            "Output frames : %d..%d (Visual Editor chooses K)"
            % dataset.settings.frame_selection_bounds
        )
        print("Output dir    : %s" % output_dir)
        print(
            "Victim run    : %s"
            % ("enabled" if args.enable_victim_validation else "disabled")
        )
        print("%s\n" % ("=" * 80))

    progress = ProgressReporter(output_dir, quiet=args.quiet)
    plan = generate_montage_plan(
        dataset=dataset,
        settings=settings,
        agents_config=agents_config,
        output_dir=str(output_dir),
        model_overrides={
            "screenwriter": args.screenwriter_model,
            "visual_editor": args.visual_editor_model,
            "caption_editor": args.caption_editor_model,
            "director": args.director_model,
        },
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        timeout_seconds=args.api_timeout_seconds,
        max_retries=args.api_max_retries,
        progress_callback=progress,
        verbose=verbose,
    )
    plan_path = save_montage_plan(plan, str(output_dir))

    materialized = {"skipped": True}
    distributed_publication = {"skipped": True}
    if not args.no_materialize:
        materialized = ImagePublisher().materialize(plan, str(output_dir))
        distributed_publication = DistributedImagePublisher().publish(
            sequence_path=materialized["captioned_image_sequence"],
            output_dir=str(output_dir),
            account_ids=dataset.settings.account_ids,
            start_time=dataset.settings.start_time,
            interval_minutes=dataset.settings.interval_minutes,
        )
        materialized["distributed_publication"] = distributed_publication
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        payload["materialized"] = materialized
        payload["distributed_publication"] = distributed_publication
        plan_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    victim_validation = "not_run"
    if args.enable_victim_validation:
        validation_config = settings.get("victim_validation") or {}
        num_victims, victim_model_names = resolve_victim_configuration(
            args.victim_models,
            args.num_victims,
            int(validation_config.get("num_victims", 5)),
        )
        num_conclusions = (
            args.num_victim_conclusions
            if args.num_victim_conclusions is not None
            else int(validation_config.get("num_conclusions", 1))
        )
        ground_truth = args.target_ground_truth or str(
            validation_config.get("ground_truth", "false")
        )
        confidence_threshold = (
            args.victim_confidence_threshold
            if args.victim_confidence_threshold is not None
            else float(
                validation_config.get("high_confidence_threshold", 0.8)
            )
        )
        progress(
            {
                "event": "victim_validation_start",
                "num_victims": num_victims,
            }
        )
        if victim_model_names:
            victim_client = [
                build_validation_client(
                    settings,
                    "victim",
                    model_override=model_name,
                    api_base=args.api_base,
                    api_key_env=args.api_key_env,
                    timeout_seconds=args.api_timeout_seconds,
                    max_retries=args.api_max_retries,
                )
                for model_name in victim_model_names
            ]
        else:
            victim_client = build_validation_client(
                settings,
                "victim",
                model_override=args.victim_model,
                api_base=args.api_base,
                api_key_env=args.api_key_env,
                timeout_seconds=args.api_timeout_seconds,
                max_retries=args.api_max_retries,
            )
        judge_client = None
        if not args.skip_ai_judge:
            judge_client = build_validation_client(
                settings,
                "judge",
                model_override=args.judge_model,
                api_base=args.api_base,
                api_key_env=args.api_key_env,
                timeout_seconds=args.api_timeout_seconds,
                max_retries=args.api_max_retries,
            )
        validation_result = run_victim_validation(
            sequence=materialized["captioned_image_sequence"],
            target_conclusion=dataset.target.label,
            ground_truth=ground_truth,
            num_victims=num_victims,
            victim_client=victim_client,
            num_conclusions=num_conclusions,
            high_confidence_threshold=confidence_threshold,
            judge_client=judge_client,
            victim_system_prompt=validation_prompt(agents_config, "victim"),
            judge_system_prompt=validation_prompt(agents_config, "judge"),
            use_cot=not args.victim_no_cot,
            verbose=verbose,
        )
        validation_path = save_victim_validation(
            validation_result, output_dir / "victim_validation.json"
        )
        victim_validation = {
            "performed": True,
            "status": "complete",
            "result_file": str(validation_path),
            "analysis": validation_result["analysis"],
            "adjudication": validation_result["adjudication"],
        }
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        payload["victim_validation"] = victim_validation
        plan_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        progress(
            {
                "event": "victim_validation_complete",
                "result": str(validation_path),
            }
        )
    provenance = write_attack_plan_provenance(
        plan,
        str(output_dir),
        materialized=materialized,
        distributed_publication=distributed_publication,
        victim_validation=victim_validation,
    )
    progress(
        {
            "event": "provenance_complete",
            "markdown": provenance["markdown"],
            "json": provenance["json"],
        }
    )
    progress({"event": "run_complete", "plan": str(plan_path)})

    print(
        json.dumps(
            {
                "status": "accepted",
                "output": str(plan_path),
                "selected_units": plan.visual_sequence.ordered_unit_ids,
                "ordered_units": plan.visual_sequence.ordered_unit_ids,
                "agent_models": plan.agent_models,
                "director_gate": plan.caption_sequence_decision.to_dict(),
                "materialized": materialized,
                "distributed_publication": distributed_publication,
                "victim_validation": victim_validation,
                "provenance": provenance,
                "research_only": True,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MontageError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
