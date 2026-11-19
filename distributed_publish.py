#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from src.environment import DistributedImagePublisher
from src.utils.data_loader import MontageError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a validation-compatible local publication manifest"
    )
    parser.add_argument(
        "--sequence",
        required=True,
        help="Path to sequence/captioned_image_sequence.json",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--account-ids",
        default=None,
        help="Optional comma-separated publisher IDs; existing IDs are preserved by default",
    )
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--interval-minutes", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    account_ids = None
    if args.account_ids:
        account_ids = [value.strip() for value in args.account_ids.split(",")]
    try:
        result = DistributedImagePublisher().publish(
            sequence_path=args.sequence,
            output_dir=args.output_dir,
            account_ids=account_ids,
            start_time=args.start_time,
            interval_minutes=args.interval_minutes,
        )
    except MontageError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
