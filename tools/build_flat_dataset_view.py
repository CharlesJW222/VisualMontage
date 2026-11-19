#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATEGORY_ROOT = ROOT / "Video_Dataset_Categories"
VIEW_ROOT = ROOT / "Video Datasets"
INDEX_SCHEMA = "visual_montage.dataset_index.v1"


def categorized_datasets():
    found = []
    for category in sorted(path for path in CATEGORY_ROOT.iterdir() if path.is_dir()):
        for directory in sorted(path for path in category.iterdir() if path.is_dir()):
            if (directory / "dataset.json").is_file():
                found.append((category.name, directory))
    return found


def clean() -> int:
    if not VIEW_ROOT.exists():
        return 0
    removed = 0
    for entry in VIEW_ROOT.iterdir():
        if entry.is_symlink() or entry.name == "index.json":
            entry.unlink()
            removed += 1
    try:
        VIEW_ROOT.rmdir()
    except OSError:
        print("Kept %s: it still contains entries not created by this tool" % VIEW_ROOT)
    return removed


def build() -> int:
    datasets = categorized_datasets()
    if not datasets:
        raise SystemExit("No dataset.json found under %s" % CATEGORY_ROOT)
    names = [directory.name for _category, directory in datasets]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SystemExit("Dataset directory names must be unique: %s" % duplicates)

    clean()
    VIEW_ROOT.mkdir(parents=True, exist_ok=True)
    entries = []
    for position, (category, directory) in enumerate(datasets, start=1):
        manifest = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
        link = VIEW_ROOT / directory.name
        target = os.path.relpath(directory, VIEW_ROOT)
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            raise SystemExit(
                "Cannot create symbolic link %s -> %s (%s). On Windows, enable "
                "Developer Mode or run from an elevated shell." % (link, target, exc)
            )
        entries.append(
            {
                "index": position,
                "directory": directory.name,
                "dataset_id": manifest.get("dataset_id", directory.name),
                "category": category,
                "ground_truth": manifest.get("ground_truth", False),
            }
        )
    index = {
        "schema_version": INDEX_SCHEMA,
        "data_root": ".",
        "dataset_count": len(entries),
        "datasets": entries,
    }
    (VIEW_ROOT / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return len(entries)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the flat Video Datasets/ view of the categorized corpus."
    )
    parser.add_argument("--clean", action="store_true", help="remove the flat view")
    args = parser.parse_args()
    if args.clean:
        print("Removed %d entries from %s" % (clean(), VIEW_ROOT))
        return 0
    count = build()
    print("Linked %d datasets under %s" % (count, VIEW_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
