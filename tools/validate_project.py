#!/usr/bin/env python3
import json
import re
import sys
from dataclasses import fields
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import ProductionSettings, load_dataset  # noqa: E402
from scripts.run_all_datasets import load_index  # noqa: E402


ROLES = ("screenwriter", "visual_editor", "caption_editor", "director")
VALIDATION_ROLES = ("victim", "judge")
REQUIRED_MODULES = (
    "src/production/screenwriter.py",
    "src/production/visual_editor.py",
    "src/production/caption_editor.py",
    "src/production/director.py",
    "generate_montage_plan.py",
    "demo.py",
    "victim_validation.py",
)
FORBIDDEN_KEYS = {
    "nodes",
    "edges",
    "source",
    "source_graph",
    "supported_edges",
    "delta_edges",
    "seed_beliefs",
    "singleton_beliefs",
    "captions",
}
FRAME_RECORD_KEYS = {"id", "image_path", "image_sha256"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SETTINGS_KEYS = {item.name for item in fields(ProductionSettings)}
ADVISORY_SETTINGS_KEYS = {
    "recommended_frame_interval_seconds",
    "recommended_candidate_frame_count",
}


def collect_keys(value):
    keys = set()
    if isinstance(value, dict):
        keys.update(value.keys())
        for item in value.values():
            keys.update(collect_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(collect_keys(item))
    return keys


def collect_strings(value):
    strings = set()
    if isinstance(value, str):
        strings.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            strings.update(collect_strings(item))
    elif isinstance(value, list):
        for item in value:
            strings.update(collect_strings(item))
    return strings


def main():
    for relative in REQUIRED_MODULES:
        if not (ROOT / relative).is_file():
            raise ValueError("Missing module: %s" % relative)
    settings = json.loads((ROOT / "configs/settings.json").read_text("utf-8"))
    agents = json.loads((ROOT / "configs/agents.json").read_text("utf-8"))
    role_configs = (settings.get("llm") or {}).get("roles") or {}
    prompts = agents.get("production_team") or {}
    for role in ROLES:
        if not str((role_configs.get(role) or {}).get("model") or "").strip():
            raise ValueError("Missing model for %s" % role)
        if not str((prompts.get(role) or {}).get("system_prompt") or "").strip():
            raise ValueError("Missing prompt for %s" % role)
    validation_prompts = agents.get("validation_team") or {}
    for role in VALIDATION_ROLES:
        if not str((role_configs.get(role) or {}).get("model") or "").strip():
            raise ValueError("Missing model for %s" % role)
        if not str(
            (validation_prompts.get(role) or {}).get("system_prompt") or ""
        ).strip():
            raise ValueError("Missing prompt for %s" % role)

    index_path = ROOT / "Video Datasets" / "index.json"
    indexed_items = load_index(index_path)
    index_payload = json.loads(index_path.read_text("utf-8"))
    dataset_root = (index_path.parent / index_payload.get("data_root", ".")).resolve()
    if index_payload.get("schema_version") != "visual_montage.dataset_index.v1":
        raise ValueError("Video Datasets/index.json schema_version is inconsistent")
    indexed_directories = [
        str(item.get("directory")) for item in index_payload.get("datasets") or []
    ]
    if index_payload.get("dataset_count") != len(indexed_directories):
        raise ValueError("Video Datasets/index.json dataset_count is inconsistent")
    dataset_dirs = [dataset_root / directory for directory in indexed_directories]
    missing_directories = [path.name for path in dataset_dirs if not path.is_dir()]
    if missing_directories:
        raise ValueError(
            "Video Datasets/index.json references missing directories: %s"
            % missing_directories
        )
    actual_directories = [
        path.name
        for path in sorted(
            (
                path
                for path in dataset_root.iterdir()
                if path.is_dir() and (path / "dataset.json").is_file()
            ),
            key=lambda path: indexed_directories.index(path.name)
            if path.name in indexed_directories
            else len(indexed_directories),
        )
    ]
    if indexed_directories != actual_directories:
        raise ValueError("Video Datasets/index.json directory order does not match existing dataset directories")

    reports = []
    for index_item, directory in zip(indexed_items, dataset_dirs):
        manifest = directory / "dataset.json"
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError("%s requires metadata.json" % directory)
        raw = json.loads(manifest.read_text("utf-8"))
        metadata = json.loads(metadata_path.read_text("utf-8"))
        if metadata.get("schema_version") != "visual_montage.dataset_metadata.v1":
            raise ValueError("%s metadata schema is inconsistent" % metadata_path)
        if metadata.get("ground_truth") is not False:
            raise ValueError("%s ground_truth must be false" % metadata_path)
        if metadata.get("target_source") != "dataset.json":
            raise ValueError("%s must point target_source to dataset.json" % metadata_path)
        target_labels = {
            str(item.get("label"))
            for item in [raw.get("target")] + list(raw.get("targets") or [])
            if isinstance(item, dict) and item.get("label")
        }
        if "target" in metadata or (
            target_labels & collect_strings(metadata.get("target_design"))
        ):
            raise ValueError("%s must not duplicate the target" % metadata_path)
        if metadata.get("dataset_id") != raw.get("dataset_id"):
            raise ValueError("%s dataset_id does not match dataset.json" % metadata_path)
        if index_item.get("dataset_id", raw.get("dataset_id")) != raw.get("dataset_id"):
            raise ValueError("%s dataset_id does not match the index" % manifest)
        video_file = raw.get("video_file")
        is_video = video_file is not None
        if is_video:
            if not isinstance(video_file, str) or not video_file.strip():
                raise ValueError("%s video_file must be a non-empty path" % manifest)
            video = (directory / Path(video_file).expanduser()).resolve()
            if not video.is_file() or video.stat().st_size == 0:
                raise ValueError("Video file is missing or empty: %s" % video)
            raw_sequence = {}
        else:
            sequence = directory / raw.get("image_sequence_file", "image_sequence.json")
            raw_sequence = json.loads(sequence.read_text("utf-8"))
        forbidden = sorted(
            (collect_keys(raw) | collect_keys(raw_sequence)) & FORBIDDEN_KEYS
        )
        if forbidden:
            raise ValueError("%s contains legacy keys: %s" % (manifest, forbidden))
        settings_value = raw.get("settings") or {}
        if set(settings_value) - SETTINGS_KEYS - ADVISORY_SETTINGS_KEYS:
            raise ValueError("%s contains unsupported settings" % manifest)
        dataset = load_dataset(
            str(manifest), load_images=not is_video,
            settings_defaults=settings.get("production"),
        )
        if is_video:
            reports.append({
                "index": int(index_item.get("index")),
                "directory": directory.name,
                "dataset_id": dataset.dataset_id,
                "input_type": "video",
                "video_file": str(video),
                "target_count": len(dataset.targets),
                "min_output_frames": dataset.settings.frame_selection_bounds[0],
                "max_output_frames": dataset.settings.frame_selection_bounds[1],
            })
            continue
        frame_records = raw_sequence.get("image_sequence")
        if not isinstance(frame_records, list) or not frame_records:
            raise ValueError("%s requires image_sequence records" % sequence)
        expected_ids = [
            "frame_%03d" % index for index in range(1, len(frame_records) + 1)
        ]
        actual_ids = []
        referenced_images = []
        for record in frame_records:
            if not isinstance(record, dict) or set(record) != FRAME_RECORD_KEYS:
                raise ValueError(
                    "%s must use only id, image_path, and image_sha256" % sequence
                )
            frame_id = str(record.get("id") or "")
            if not re.fullmatch(r"frame_\d{3}", frame_id):
                raise ValueError("Invalid neutral frame ID: %s" % frame_id)
            actual_ids.append(frame_id)
            referenced_images.append(str(record["image_path"]))
        if actual_ids != expected_ids:
            raise ValueError("%s frame IDs are not sequential" % sequence)
        media_files = sorted(
            path.relative_to(directory).as_posix()
            for path in (directory / "images").iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if sorted(referenced_images) != media_files:
            raise ValueError("%s has missing or unreferenced image files" % directory)
        reports.append(
            {
                "index": int(index_item.get("index")),
                "directory": directory.name,
                "dataset_id": dataset.dataset_id,
                "frame_nodes": len(dataset.image_units),
                "images": len(dataset.image_units),
                "output_frames": dataset.settings.output_frames,
            }
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "agents": list(ROLES),
                "validation_agents": list(VALIDATION_ROLES),
                "modules": list(REQUIRED_MODULES),
                "datasets": reports,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
