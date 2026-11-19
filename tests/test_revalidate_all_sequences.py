import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "revalidate_all_sequences.py"
)
SPEC = importlib.util.spec_from_file_location("revalidate_all_sequences", SCRIPT_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(RUNNER)


class RevalidateAllSequencesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "Video Datasets").mkdir()
        (self.root / "Video Datasets" / "index.json").write_text(
            json.dumps(
                {
                    "datasets": [
                        {"directory": "complete", "ground_truth": "false"},
                        {"directory": "missing", "ground_truth": "true"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.output_root = self.root / "attack_plan" / "run"
        sequence = self.output_root / "complete" / "sequence"
        sequence.mkdir(parents=True)
        (sequence / "captioned_image_sequence.json").write_text(
            "{}", encoding="utf-8"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_dry_run_records_command_and_missing_sequences(self):
        args = RUNNER.build_parser().parse_args(
            [
                "--output-root",
                "attack_plan/run",
                "--dry-run",
            ]
        )
        self.assertEqual(RUNNER.run_revalidation(args, self.root), 0)

        records = [
            json.loads(line)
            for line in (self.output_root / "revalidation_gpt54_status.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        by_dataset = {record["dataset"]: record for record in records}
        command = by_dataset["complete"]["command"]
        self.assertEqual(by_dataset["complete"]["status"], "dry_run")
        self.assertIn("gpt-5.4", command)
        self.assertIn("--judge-model", command)
        self.assertEqual(
            by_dataset["missing"]["status"], "skipped_missing_sequence"
        )

    def test_resume_skips_existing_named_result(self):
        result = self.output_root / "complete" / "victim_validation_gpt54.json"
        result.write_text(
            json.dumps(
                {
                    "analysis": {"ASR": 40.0},
                    "adjudication": {"ai_judge": {"decision": "accept"}},
                }
            ),
            encoding="utf-8",
        )
        args = RUNNER.build_parser().parse_args(
            [
                "--output-root",
                "attack_plan/run",
                "--datasets",
                "complete",
                "--resume",
            ]
        )
        self.assertEqual(RUNNER.run_revalidation(args, self.root), 0)
        record = json.loads(
            (self.output_root / "revalidation_gpt54_status.jsonl")
            .read_text(encoding="utf-8")
            .strip()
        )
        self.assertEqual(record["status"], "skipped_existing_result")
        self.assertEqual(record["metrics"], {"ASR": 40.0})
        self.assertEqual(record["ai_judge"]["decision"], "accept")

    def test_result_name_must_stay_inside_each_dataset_directory(self):
        with self.assertRaisesRegex(ValueError, "plain .json filename"):
            RUNNER.validate_result_name("nested/result.json")


if __name__ == "__main__":
    unittest.main()
