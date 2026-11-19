from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..utils.data_loader import MontageError


DEFAULT_ACCOUNT_IDS = (
    "research_account_001",
    "research_account_002",
)


class DistributedImagePublisher:
    SCHEMA_VERSION = "visual_montage.simulated_distributed_publication.v1"

    def publish(
        self,
        sequence_path: str,
        output_dir: Optional[str] = None,
        account_ids: Optional[Sequence[str]] = None,
        start_time: Optional[str] = None,
        interval_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        sequence_file = Path(sequence_path).expanduser().resolve()
        if not sequence_file.is_file():
            raise MontageError(
                "Materialized sequence not found: %s" % sequence_file
            )

        try:
            payload = json.loads(sequence_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MontageError(
                "Invalid materialized sequence JSON: %s" % sequence_file
            ) from exc

        if not isinstance(payload, dict):
            raise MontageError("Materialized sequence must contain an object")
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            raise MontageError("Materialized sequence requires an items array")

        accounts = self._resolve_accounts(raw_items, account_ids)
        interval = self._resolve_interval(interval_minutes)
        base_time = self._resolve_start_time(
            raw_items, start_time, interval
        )

        records: List[Dict[str, Any]] = []
        channel_items: Dict[str, List[str]] = {
            account_id: [] for account_id in accounts
        }

        for position, raw_item in enumerate(raw_items, start=1):
            if not isinstance(raw_item, Mapping):
                raise MontageError(
                    "Materialized sequence item %d must be an object" % position
                )

            publisher_id = str(raw_item.get("publisher_id") or "").strip()
            if not publisher_id:
                publisher_id = accounts[(position - 1) % len(accounts)]
            if publisher_id not in channel_items:
                channel_items[publisher_id] = []

            timestamp = str(raw_item.get("display_timestamp") or "").strip()
            if not timestamp:
                timestamp = (
                    base_time + timedelta(minutes=(position - 1) * interval)
                ).isoformat()

            publication_id = "publication_%04d" % position
            channel_id = "public_channel_%s" % publisher_id
            channel_items[publisher_id].append(publication_id)
            records.append(
                {
                    "publication_id": publication_id,
                    "sequence_position": position,
                    "feed_item_id": raw_item.get(
                        "feed_item_id", "montage_%04d" % position
                    ),
                    "publisher_id": publisher_id,
                    "channel_id": channel_id,
                    "display_timestamp": timestamp,
                    "media_type": raw_item.get("media_type"),
                    "caption": raw_item.get("caption"),
                    "image_path": raw_item.get("image_path"),
                    "unit_id": raw_item.get("unit_id"),
                    "publication_state": "simulated_ready",
                }
            )

        for raw_item, record in zip(raw_items, records):
            raw_item["publisher_id"] = record["publisher_id"]
            raw_item["display_timestamp"] = record["display_timestamp"]
            raw_item["publication_id"] = record["publication_id"]
            raw_item["channel_id"] = record["channel_id"]
            raw_item["distribution_mode"] = "simulated_public_channel"
            raw_item["publication_state"] = record["publication_state"]

        channels = [
            {
                "channel_id": "public_channel_%s" % account_id,
                "publisher_id": account_id,
                "publication_count": len(publication_ids),
                "publication_ids": publication_ids,
            }
            for account_id, publication_ids in channel_items.items()
        ]
        summary = {
            "schema_version": self.SCHEMA_VERSION,
            "mode": "simulated_distributed_publication",
            "external_publication_performed": False,
            "research_only": True,
            "publication_allowed": False,
            "total_items": len(records),
            "channel_count": len(channels),
            "accounts": [channel["publisher_id"] for channel in channels],
            "interval_minutes": interval,
        }

        sequence_root = sequence_file.parent
        manifest_path = sequence_root / "distributed_publication.json"
        manifest = {
            **summary,
            "source_sequence": str(sequence_file),
            "target": payload.get("target"),
            "channels": channels,
            "items": records,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        summary = {
            **summary,
            "manifest_path": str(manifest_path),
            "sequence_path": str(sequence_file),
        }
        payload["distributed_publication"] = summary
        sequence_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        plan_root = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else sequence_root.parent
        )
        plan_path = plan_root / "montage_plan.json"
        if plan_path.is_file():
            try:
                plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise MontageError("Invalid montage plan JSON: %s" % plan_path) from exc
            if not isinstance(plan_payload, dict):
                raise MontageError("Montage plan must contain an object")
            summary["plan_path"] = str(plan_path)
            plan_payload["distributed_publication"] = summary
            stages = plan_payload.setdefault("stages", {})
            stage7 = stages.setdefault("stage_7_distributed_publication", {})
            stage7["simulation"] = summary
            stage7["external_publication_performed"] = False
            plan_path.write_text(
                json.dumps(plan_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return summary

    @staticmethod
    def _resolve_accounts(
        items: Sequence[Any], account_ids: Optional[Sequence[str]]
    ) -> List[str]:
        supplied = [str(value).strip() for value in (account_ids or ())]
        supplied = [value for value in supplied if value]
        if supplied:
            return list(dict.fromkeys(supplied))

        existing: List[str] = []
        for item in items:
            if isinstance(item, Mapping):
                account_id = str(item.get("publisher_id") or "").strip()
                if account_id and account_id not in existing:
                    existing.append(account_id)
        return existing or list(DEFAULT_ACCOUNT_IDS)

    @staticmethod
    def _resolve_interval(interval_minutes: Optional[int]) -> int:
        if interval_minutes is None:
            return 10
        try:
            interval = int(interval_minutes)
        except (TypeError, ValueError) as exc:
            raise MontageError("interval_minutes must be an integer") from exc
        if interval < 0:
            raise MontageError("interval_minutes must be >= 0")
        return interval

    @staticmethod
    def _resolve_start_time(
        items: Sequence[Any],
        start_time: Optional[str],
        interval_minutes: int,
    ) -> datetime:
        candidate = str(start_time or "").strip()
        if not candidate:
            for item in items:
                if isinstance(item, Mapping) and item.get("display_timestamp"):
                    candidate = str(item["display_timestamp"]).strip()
                    break
        if not candidate:
            return datetime.now().astimezone()
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError as exc:
            raise MontageError(
                "Invalid publication start timestamp: %s" % candidate
            ) from exc


__all__ = ["DistributedImagePublisher"]
