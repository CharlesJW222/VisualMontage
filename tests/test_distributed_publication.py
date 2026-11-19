import json
import tempfile
import unittest
from pathlib import Path

from src.environment import DistributedImagePublisher
from victim_validation import build_multimodal_victim_feed


class DistributedPublicationTests(unittest.TestCase):
    def test_adds_channels_without_changing_validation_visible_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence_dir = root / "sequence"
            sequence_dir.mkdir()
            image_path = root / "frame.png"
            image_path.write_bytes(b"placeholder")
            sequence_path = sequence_dir / "captioned_image_sequence.json"
            sequence_path.write_text(
                json.dumps(
                    {
                        "target": {"label": "test target"},
                        "items": [
                            {
                                "feed_item_id": "montage_0001",
                                "sequence_position": 1,
                                "publisher_id": "account_a",
                                "display_timestamp": "2026-01-01T10:00:00+08:00",
                                "image_path": str(image_path),
                                "caption": None,
                            },
                            {
                                "feed_item_id": "montage_0002",
                                "sequence_position": 2,
                                "publisher_id": "account_b",
                                "display_timestamp": "2026-01-01T10:10:00+08:00",
                                "image_path": None,
                                "caption": "A visible caption.",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "montage_plan.json").write_text(
                json.dumps(
                    {"stages": {"stage_7_distributed_publication": {"psi": []}}}
                ),
                encoding="utf-8",
            )

            result = DistributedImagePublisher().publish(
                str(sequence_path),
                output_dir=str(root),
                account_ids=("account_a", "account_b"),
                interval_minutes=10,
            )

            self.assertEqual(result["channel_count"], 2)
            self.assertEqual(result["total_items"], 2)
            self.assertTrue(Path(result["manifest_path"]).is_file())
            self.assertTrue(Path(result["plan_path"]).is_file())

            updated = json.loads(sequence_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["publisher_id"] for item in updated["items"]],
                ["account_a", "account_b"],
            )
            self.assertEqual(
                [item["display_timestamp"] for item in updated["items"]],
                [
                    "2026-01-01T10:00:00+08:00",
                    "2026-01-01T10:10:00+08:00",
                ],
            )
            self.assertEqual(
                updated["distributed_publication"]["mode"],
                "simulated_distributed_publication",
            )

            feed = build_multimodal_victim_feed(sequence_path)
            self.assertEqual(
                [item["publisher_id"] for item in feed["items"]],
                ["account_a", "account_b"],
            )
            self.assertEqual(feed["items"][1]["text"], "A visible caption.")


if __name__ == "__main__":
    unittest.main()
