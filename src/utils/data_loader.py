from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


class MontageError(ValueError):
    pass
def required_text(value: Any, field_name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise MontageError("%s must not be empty" % field_name)
    return result


def required_english_text(value: Any, field_name: str) -> str:
    result = required_text(value, field_name)
    if any(character.isalpha() and not character.isascii() for character in result):
        raise MontageError(
            "%s must be written in English using Latin text" % field_name
        )
    return result


def positive_int(value: Any, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise MontageError("%s must be an integer" % field_name) from exc
    if result < 1:
        raise MontageError("%s must be >= 1" % field_name)
    return result


def probability(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MontageError("%s must be a number" % field_name) from exc
    if result < 0.0 or result > 1.0:
        raise MontageError("%s must be between 0 and 1" % field_name)
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Target:
    target_id: str
    label: str

    def to_dict(self) -> Dict[str, str]:
        return {"id": self.target_id, "label": self.label}


@dataclass
class ImageUnit:
    unit_id: str
    image_paths: List[str]
    image_sha256: List[str]
    timestamp_seconds: Optional[float] = None
    source_video: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for name in ("timestamp_seconds", "source_video"):
            if result[name] is None:
                result.pop(name)
        return result


@dataclass
class ProductionSettings:
    screenwriter_rounds: int = 3
    visual_editor_rounds: int = 3
    caption_editor_rounds: int = 3
    output_frames: Optional[int] = None
    tau: float = 0.80
    epsilon: float = 0.05
    search_alpha: float = 0.10
    search_beta: float = 0.05
    json_retries: int = 1
    account_ids: Tuple[str, ...] = (
        "research_account_001",
        "research_account_002",
    )
    start_time: str = "2026-08-07T10:00:00+08:00"
    interval_minutes: int = 10
    min_output_frames: Optional[int] = None
    max_output_frames: Optional[int] = None

    def __post_init__(self) -> None:
        if self.min_output_frames is None and self.max_output_frames is None:
            if self.output_frames is not None:
                self.output_frames = positive_int(
                    self.output_frames, "settings.output_frames"
                )
                return
        self.min_output_frames = positive_int(
            5 if self.min_output_frames is None else self.min_output_frames,
            "settings.min_output_frames",
        )
        self.max_output_frames = positive_int(
            30 if self.max_output_frames is None else self.max_output_frames,
            "settings.max_output_frames",
        )
        if not 5 <= self.min_output_frames <= self.max_output_frames <= 30:
            raise MontageError(
                "settings require 5 <= min_output_frames <= max_output_frames <= 30"
            )
        self.output_frames = None

    @classmethod
    def from_dict(
        cls, value: Optional[Mapping[str, Any]]
    ) -> "ProductionSettings":
        data = dict(value or {})
        unsupported = {"visual_editor_beam_width", "relational_graph_workers"} & data.keys()
        if unsupported:
            raise MontageError(
                "Unsupported settings: %s" % ", ".join(sorted(unsupported))
            )
        accounts = data.get("account_ids") or list(cls.account_ids)
        settings = cls(
            screenwriter_rounds=positive_int(
                data.get("screenwriter_rounds", 3),
                "settings.screenwriter_rounds",
            ),
            visual_editor_rounds=positive_int(
                data.get("visual_editor_rounds", 3),
                "settings.visual_editor_rounds",
            ),
            caption_editor_rounds=positive_int(
                data.get("caption_editor_rounds", 3),
                "settings.caption_editor_rounds",
            ),
            output_frames=data.get("output_frames"),
            min_output_frames=data.get("min_output_frames"),
            max_output_frames=data.get("max_output_frames"),
            tau=probability(data.get("tau", 0.80), "settings.tau"),
            epsilon=probability(
                data.get("epsilon", 0.05), "settings.epsilon"
            ),
            search_alpha=probability(
                data.get("search_alpha", 0.10), "settings.search_alpha"
            ),
            search_beta=probability(
                data.get("search_beta", 0.05), "settings.search_beta"
            ),
            json_retries=int(data.get("json_retries", 1)),
            account_ids=tuple(
                required_text(item, "settings.account_ids[]")
                for item in accounts
            ),
            start_time=required_text(
                data.get("start_time", cls.start_time), "settings.start_time"
            ),
            interval_minutes=int(data.get("interval_minutes", 10)),
        )
        if settings.json_retries < 0:
            raise MontageError("settings.json_retries must be >= 0")
        if not settings.account_ids:
            raise MontageError("settings.account_ids must not be empty")
        if settings.interval_minutes < 0:
            raise MontageError("settings.interval_minutes must be >= 0")
        return settings

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["account_ids"] = list(self.account_ids)
        if self.output_frames is None:
            result.pop("output_frames")
        else:
            result.pop("min_output_frames")
            result.pop("max_output_frames")
        return result

    @property
    def frame_selection_bounds(self) -> Tuple[int, int]:
        if self.output_frames is not None:
            return self.output_frames, self.output_frames
        return self.min_output_frames, self.max_output_frames

    @property
    def frame_selection_count(self) -> int:
        return self.frame_selection_bounds[0]


@dataclass
class MontageDataset:
    dataset_id: str
    root: str
    target: Target
    image_units: List[ImageUnit]
    settings: ProductionSettings
    targets: Optional[List[Target]] = None

    def __post_init__(self) -> None:
        if self.targets is None:
            self.targets = [self.target]
        if not self.targets:
            raise MontageError("dataset requires at least one target")
        target_ids = [item.target_id for item in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise MontageError("targets must have unique ids")
        if self.target.target_id not in set(target_ids):
            raise MontageError("active target must be present in targets")

    @property
    def target_ids(self) -> List[str]:
        return [item.target_id for item in self.targets or []]

    def select_target(self, target_id: str) -> "MontageDataset":
        requested = required_text(target_id, "target_id")
        for target in self.targets or []:
            if target.target_id == requested:
                return replace(self, target=target)
        raise MontageError(
            "Unknown target_id %s; available targets: %s"
            % (requested, ", ".join(self.target_ids))
        )


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MontageError("Cannot read JSON file %s: %s" % (path, exc)) from exc
    except json.JSONDecodeError as exc:
        raise MontageError("Invalid JSON file %s: %s" % (path, exc)) from exc
    if not isinstance(value, dict):
        raise MontageError("JSON file must contain an object: %s" % path)
    return value


def resolve_dataset_file(path: str) -> Path:
    value = Path(path).expanduser().resolve()
    if value.is_dir():
        value = value / "dataset.json"
    if not value.is_file():
        raise MontageError("dataset.json not found: %s" % value)
    return value


def load_dataset(
    path: str, *, load_images: bool = True, validate_frame_count: bool = True,
    settings_defaults: Optional[Mapping[str, Any]] = None,
) -> MontageDataset:
    manifest_path = resolve_dataset_file(path)
    root = manifest_path.parent.resolve()
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "visual_montage.target_image_sequence.v1":
        raise MontageError(
            "Unsupported schema_version; expected "
            "visual_montage.target_image_sequence.v1"
        )

    targets_value = manifest.get("targets")
    if targets_value is None:
        target_value = manifest.get("target")
        if not isinstance(target_value, Mapping):
            raise MontageError(
                "dataset.json requires a target object or targets array"
            )
        targets_value = [target_value]
    elif not isinstance(targets_value, list) or not targets_value:
        raise MontageError("dataset.targets must be a non-empty array")

    targets = []
    seen_target_ids = set()
    for index, target_value in enumerate(targets_value):
        if not isinstance(target_value, Mapping):
            raise MontageError("targets[%d] must be an object" % index)
        target = Target(
            required_text(
                target_value.get("id", "target_%03d" % (index + 1)),
                "targets[%d].id" % index,
            ),
            required_english_text(
                target_value.get("label"), "targets[%d].label" % index
            ),
        )
        if target.target_id in seen_target_ids:
            raise MontageError("Duplicate target id: %s" % target.target_id)
        seen_target_ids.add(target.target_id)
        targets.append(target)

    image_units = []
    if load_images:
        sequence_file = required_text(
            manifest.get("image_sequence_file", "image_sequence.json"),
            "image_sequence_file",
        )
        sequence_path = (root / sequence_file).resolve()
        try:
            sequence_path.relative_to(root)
        except ValueError as exc:
            raise MontageError(
                "image_sequence_file must remain inside the dataset"
            ) from exc

        sequence_manifest = load_json(sequence_path)
        raw_units = sequence_manifest.get("image_sequence")
        if not isinstance(raw_units, list) or not raw_units:
            raise MontageError(
                "image_sequence.json requires a non-empty image_sequence array"
            )

        seen = set()
        seen_frame_ids = set()
        for index, raw in enumerate(raw_units):
            if not isinstance(raw, Mapping):
                raise MontageError("image_sequence[%d] must be an object" % index)
            unexpected = sorted(
                set(raw) - {"id", "image_path", "image_sha256"}
            )
            if unexpected:
                raise MontageError(
                    "image_sequence entries use unsupported keys: %s" % unexpected
                )
            unit_id = required_text(raw.get("id"), "image_sequence[].id")
            if not re.fullmatch(r"frame_\d{3}", unit_id):
                raise MontageError(
                    "image_sequence IDs must use frame_###: %s" % unit_id
                )
            if unit_id in seen:
                raise MontageError("Duplicate image unit ID: %s" % unit_id)
            seen.add(unit_id)
            relative_path = required_text(
                raw.get("image_path"), "image_sequence[].image_path"
            )
            expected_hash = required_text(
                raw.get("image_sha256"), "image_sequence[].image_sha256"
            ).lower()
            image_path = (root / relative_path).resolve()
            try:
                image_path.relative_to(root)
            except ValueError as exc:
                raise MontageError("Image path must remain inside dataset") from exc
            if not image_path.is_file():
                raise MontageError("Image file not found: %s" % image_path)
            actual_hash = file_sha256(image_path)
            if actual_hash != expected_hash:
                raise MontageError("Image SHA-256 mismatch: %s" % image_path)
            if unit_id in seen_frame_ids:
                raise MontageError("Duplicate frame ID: %s" % unit_id)
            seen_frame_ids.add(unit_id)
            image_units.append(
                ImageUnit(
                    unit_id=unit_id,
                    image_paths=[str(image_path)],
                    image_sha256=[actual_hash],
                )
            )

    manifest_settings = manifest.get("settings") or {}
    settings_values = dict(settings_defaults or {})
    if "output_frames" in manifest_settings and not any(
        name in manifest_settings for name in ("min_output_frames", "max_output_frames")
    ):
        settings_values.pop("min_output_frames", None)
        settings_values.pop("max_output_frames", None)
    settings_values.update(manifest_settings)
    settings = ProductionSettings.from_dict(settings_values)
    if load_images and validate_frame_count and settings.frame_selection_bounds[0] > len(image_units):
        raise MontageError(
            "settings frame selection count exceeds the image-frame count"
        )
    return MontageDataset(
        dataset_id=required_text(manifest.get("dataset_id"), "dataset_id"),
        root=str(root),
        target=targets[0],
        image_units=image_units,
        settings=settings,
        targets=targets,
    )
