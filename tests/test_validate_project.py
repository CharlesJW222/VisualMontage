import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_production_team import create_dataset


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_project", PROJECT / "tools/validate_project.py"
)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class ValidateProjectTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shutil.copytree(PROJECT / "configs", self.root / "configs")
        for name in VALIDATOR.REQUIRED_MODULES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.directory = self.root / "media" / "example"
        self.directory.mkdir(parents=True)
        create_dataset(self.directory)
        (self.directory / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "visual_montage.dataset_metadata.v1",
                    "dataset_id": "test_dataset",
                    "ground_truth": False,
                    "target_source": "dataset.json",
                }
            )
        )
        index = self.root / "Video Datasets" / "index.json"
        index.parent.mkdir()
        index.write_text(
            json.dumps(
                {
                    "schema_version": "visual_montage.dataset_index.v1",
                    "data_root": "../media",
                    "dataset_count": 1,
                    "datasets": [
                        {
                            "index": 1,
                            "directory": "example",
                            "dataset_id": "test_dataset",
                        }
                    ],
                }
            )
        )

    def validate(self):
        output = io.StringIO()
        with patch.object(VALIDATOR, "ROOT", self.root):
            with contextlib.redirect_stdout(output):
                self.assertEqual(VALIDATOR.main(), 0)
        return json.loads(output.getvalue())

    def make_video_manifest(self):
        path = self.directory / "dataset.json"
        manifest = json.loads(path.read_text())
        manifest["video_file"] = "source.mp4"
        manifest["settings"].pop("output_frames")
        manifest["settings"].update(min_output_frames=5, max_output_frames=10)
        path.write_text(json.dumps(manifest))
        (self.directory / "image_sequence.json").unlink()
        (self.directory / "source.mp4").write_bytes(b"video-file-presence-fixture")

    def test_image_index_still_validates_frames_and_hashes(self):
        report = self.validate()
        self.assertEqual(report["datasets"][0]["frame_nodes"], 3)
        (self.directory / "images/frame_001.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.validate()

    def test_video_does_not_require_presampled_images(self):
        self.make_video_manifest()
        report = self.validate()["datasets"][0]
        self.assertEqual(report["input_type"], "video")
        self.assertEqual(report["min_output_frames"], 5)
        self.assertEqual(report["max_output_frames"], 10)

    def test_missing_or_empty_video_is_rejected(self):
        self.make_video_manifest()
        video = self.directory / "source.mp4"
        video.unlink()
        with self.assertRaisesRegex(ValueError, "Video file is missing or empty"):
            self.validate()
        video.touch()
        with self.assertRaisesRegex(ValueError, "Video file is missing or empty"):
            self.validate()

    def test_inconsistent_metadata_is_rejected(self):
        path = self.directory / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["dataset_id"] = "wrong"
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "dataset_id does not match"):
            self.validate()

    def test_bundled_project_is_valid(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(VALIDATOR.main(), 0)


if __name__ == "__main__":
    unittest.main()
