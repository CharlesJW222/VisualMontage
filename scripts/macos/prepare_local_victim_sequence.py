#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def resolve_visible_image(sequence_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    fallback = sequence_path.parent / "victim_images" / path.name
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"Visible image not found: {value}")


def prepare(source_path: Path, destination: Path, max_side: int, quality: int) -> None:
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    items = raw.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("captioned image sequence requires a non-empty items list")

    destination.mkdir(parents=True, exist_ok=True)
    image_root = destination / "visible_images"
    image_root.mkdir(parents=True, exist_ok=True)
    copied = []
    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"items[{position - 1}] must be an object")
        if bool(item.get("image_withheld", False)) or not item.get("image_path"):
            continue
        source_image = resolve_visible_image(source_path, str(item["image_path"]))
        output_image = image_root / f"item_{position:03d}.jpg"
        with Image.open(source_image) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            image.save(output_image, format="JPEG", quality=quality, optimize=True)
        item["image_path"] = str(output_image.resolve())
        copied.append({"position": position, "source": str(source_image), "copy": str(output_image)})

    raw["validation_input_copy"] = {
        "source_sequence": str(source_path.resolve()),
        "max_image_side": max_side,
        "jpeg_quality": quality,
        "visible_images_copied": copied,
    }
    output_sequence = destination / "captioned_image_sequence.json"
    output_sequence.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_sequence)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--quality", type=int, default=82)
    args = parser.parse_args()
    if args.max_side < 1 or args.quality < 1 or args.quality > 100:
        raise ValueError("max-side must be positive and quality must be 1–100")
    prepare(Path(args.source).resolve(), Path(args.destination).resolve(), args.max_side, args.quality)


if __name__ == "__main__":
    main()
