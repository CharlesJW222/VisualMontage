import importlib.util
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_all_datasets.py"
SPEC = importlib.util.spec_from_file_location("run_all_datasets", SCRIPT_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(RUNNER)


class RunAllDatasetsTests(unittest.TestCase):
    def test_index_entries_match_indexed_dataset_directories(self):
        root = Path(__file__).resolve().parents[1]
        index = root / "Video Datasets" / "index.json"
        payload = json.loads(index.read_text())
        entries = RUNNER.load_index(index)
        self.assertEqual(len(entries), payload["dataset_count"])
        dataset_root = (index.parent / payload.get("data_root", ".")).resolve()
        actual = {
            path.name
            for path in dataset_root.iterdir()
            if (path / "dataset.json").is_file()
        }
        self.assertEqual({entry["directory"] for entry in entries}, actual)
        for entry in entries:
            directory = Path(entry["dataset_path"])
            self.assertEqual(directory, dataset_root / entry["directory"])
            self.assertTrue((directory / "dataset.json").is_file())

    def test_index_defaults_to_its_own_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            index = Path(temporary) / "index.json"
            index.write_text(json.dumps({"datasets": [{"directory": "example"}]}))
            entries = RUNNER.load_index(index)
            self.assertEqual(
                Path(entries[0]["dataset_path"]), index.parent.resolve() / "example"
            )

    def test_invalid_index_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            index = Path(temporary) / "index.json"
            for payload in (
                [],
                {"datasets": []},
                {"datasets": [{"directory": "../outside"}]},
                {"datasets": [{"directory": "same"}, {"directory": "same"}]},
                {"data_root": "", "datasets": [{"directory": "example"}]},
            ):
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    index.write_text(json.dumps(payload))
                    RUNNER.load_index(index)

    def test_video_dry_run_uses_real_paths_without_model_calls(self):
        root = Path(__file__).resolve().parents[1]
        entries = RUNNER.load_index(root / "Video Datasets" / "index.json")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            arguments = [
                str(SCRIPT_PATH), "--dry-run", "--with-validation",
                "--output-root", str(output),
            ]
            with patch.object(sys, "argv", arguments):
                with patch.object(RUNNER.subprocess, "run") as run:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(RUNNER.main(), 0)
            run.assert_not_called()
            records = [
                json.loads(line)
                for line in (output / "run_status.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), len(entries))
            for entry, record in zip(entries, records):
                command = record["command"]
                self.assertEqual(
                    command[command.index("--dataset") + 1], entry["dataset_path"]
                )
                self.assertEqual(
                    Path(command[command.index("--output") + 1]),
                    output / entry["directory"],
                )
                self.assertEqual(
                    command[command.index("--target-ground-truth") + 1], "false"
                )

    def test_subset_selection_preserves_index_order(self):
        entries = [
            {"directory": "first"},
            {"directory": "second"},
            {"directory": "third"},
        ]
        selected = RUNNER.selected_entries(entries, "third,first")
        self.assertEqual([entry["directory"] for entry in selected], ["first", "third"])

    def test_unknown_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not present"):
            RUNNER.selected_entries([{"directory": "known"}], "missing")


class LocalBatchCompletionTests(unittest.TestCase):
    def check_complete(self, dataset, output):
        root = SCRIPT_PATH.parents[1]
        script = (root / "scripts/macos/run_all_qwen_datasets.sh").read_text()
        function = (
            "dataset_complete() {"
            + script.split("dataset_complete() {", 1)[1].split("\n}\n", 1)[0]
            + "\n}\n"
        )
        result = subprocess.run(
            [
                "bash",
                "-c",
                function + '\ndataset_complete "$1" "$2"',
                "_",
                str(dataset),
                str(output),
            ],
            env={
                **os.environ,
                "ROOT": str(root),
                "PYTHON_BIN": sys.executable,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.stderr, "")
        self.assertIn(result.returncode, (0, 1))
        return result.returncode == 0

    @staticmethod
    def write_result(directory):
        for name in (
            "montage_plan.json",
            "sequence/captioned_image_sequence.json",
            "victim_validation.json",
        ):
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")

    def test_single_target_completion_requires_all_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.json"
            dataset.write_text(
                json.dumps(
                    {
                        "schema_version": "visual_montage.target_image_sequence.v1",
                        "dataset_id": "example",
                        "target": {"id": "A", "label": "Visible evidence."},
                    }
                )
            )
            output = root / "output"
            self.assertFalse(self.check_complete(dataset, output))
            self.write_result(output)
            self.assertTrue(self.check_complete(dataset, output))
            (output / "victim_validation.json").write_text("")
            self.assertFalse(self.check_complete(dataset, output))

    def test_multi_target_completion_requires_every_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.json"
            dataset.write_text(
                json.dumps(
                    {
                        "schema_version": "visual_montage.target_image_sequence.v1",
                        "dataset_id": "example",
                        "targets": [
                            {"id": "A", "label": "First target."},
                            {"id": "B", "label": "Second target."},
                        ],
                    }
                )
            )
            output = root / "output"
            self.write_result(output / "A")
            self.assertFalse(self.check_complete(dataset, output))
            self.write_result(output / "B")
            self.assertTrue(self.check_complete(dataset, output))


if __name__ == "__main__":
    unittest.main()
