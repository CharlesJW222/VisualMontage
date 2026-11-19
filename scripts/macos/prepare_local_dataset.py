#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(
    source: Path,
    destination: Path,
    max_side: int,
    quality: int,
    max_frames: int,
) -> None:
    source = source.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
    sequence_name = str(manifest.get("image_sequence_file") or "image_sequence.json")
    sequence = json.loads((source / sequence_name).read_text(encoding="utf-8"))
    output_records = []
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    source_records = list(sequence["image_sequence"])
    if max_frames > 0 and len(source_records) > max_frames:
        selected_indices = [
            round(i * (len(source_records) - 1) / (max_frames - 1))
            for i in range(max_frames)
        ] if max_frames > 1 else [0]
        source_records = [source_records[index] for index in selected_indices]
    for record in source_records:
        source_image = (source / str(record["image_path"])).resolve()
        output_name = Path(str(record["image_path"])).with_suffix(".jpg").name
        output_image = image_dir / output_name
        with Image.open(source_image) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            image.save(output_image, format="JPEG", quality=quality, optimize=True)
        output_records.append(
            {
                "id": str(record["id"]),
                "image_path": f"images/{output_name}",
                "image_sha256": sha256(output_image),
            }
        )
    manifest["image_sequence_file"] = sequence_name
    manifest.setdefault("settings", {})["output_frames"] = min(
        int(manifest.get("settings", {}).get("output_frames", len(output_records))),
        len(output_records),
    )
    (destination / "dataset.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / sequence_name).write_text(
        json.dumps({"image_sequence": output_records}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (destination / "source_dataset.txt").write_text(str(source) + "\n", encoding="utf-8")
    print(
        f"prepared {source} -> {destination} "
        f"({len(output_records)} frames, max_side={max_side})"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--quality", type=int, default=82)
    parser.add_argument("--max-frames", type=int, default=3)
    args = parser.parse_args()
    prepare(
        Path(args.source),
        Path(args.destination),
        args.max_side,
        args.quality,
        args.max_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
