import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.utils.data_loader import MontageError, file_sha256
from src.utils.video_sampler import sample_video


def create_test_video(root, offset_seconds=0):
    frames = root / "video-source"
    frames.mkdir(exist_ok=True)
    for index in range(12):
        Image.new("RGB", (16, 16), (index * 20, 0, 0)).save(
            frames / ("input_%02d.png" % index)
        )
    video = root / ("input_%s.mkv" % offset_seconds)
    command = [
        shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-framerate", "4", "-i", str(frames / "input_%02d.png"),
    ]
    if offset_seconds:
        command += ["-vf", "setpts=PTS+%d/TB" % offset_seconds]
    command += ["-c:v", "ffv1", str(video)]
    subprocess.run(command, check=True, capture_output=True)
    return video


class VideoSamplerErrorsTests(unittest.TestCase):
    def test_invalid_intervals_fail_before_reading_video(self):
        for interval in (0, -1, float("nan"), float("inf"), -float("inf"), "bad", None):
            with self.subTest(interval=interval):
                with self.assertRaisesRegex(MontageError, "finite number > 0"):
                    sample_video("/missing/video.mp4", "/unused", interval)

    def test_missing_video_is_a_montage_error(self):
        with self.assertRaisesRegex(MontageError, "Video file not found"):
            sample_video("/missing/video.mp4", "/unused")

    def test_missing_ffmpeg_reports_optional_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "source.mp4"
            video.touch()
            with patch("src.utils.video_sampler.shutil.which", return_value=None):
                with self.assertRaisesRegex(MontageError, "requires FFmpeg"):
                    sample_video(str(video), str(Path(temporary) / "output"))

    def test_zero_frames_is_an_error_and_preserves_previous_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "source.mp4"
            video.touch()
            pool = root / "output" / "candidate_frames"
            pool.mkdir(parents=True)
            old = pool / "frame_000001.png"
            old.write_bytes(b"existing sample")
            with patch("src.utils.video_sampler.shutil.which", return_value="ffmpeg"), patch(
                "src.utils.video_sampler.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, "", ""),
            ):
                with self.assertRaisesRegex(MontageError, "zero frames"):
                    sample_video(str(video), str(pool.parent))
            self.assertEqual(old.read_bytes(), b"existing sample")


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is optional and required for real video tests")
class VideoSamplerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.video = create_test_video(self.root)
        self.output = self.root / "output"

    def test_uniform_grid_authentic_pixels_metadata_and_repeatability(self):
        units = sample_video(str(self.video), str(self.output), 0.5)
        self.assertEqual([unit.timestamp_seconds for unit in units], [0, 0.5, 1, 1.5, 2, 2.5])
        self.assertEqual([unit.unit_id for unit in units], ["frame_%06d" % n for n in range(1, 7)])
        for index, unit in enumerate(units):
            self.assertEqual(unit.source_video, str(self.video.resolve()))
            with Image.open(unit.image_paths[0]) as image:
                actual = image.getpixel((0, 0))
                expected = (index * 40, 0, 0)
                self.assertTrue(
                    all(abs(value - target) <= 3 for value, target in zip(actual, expected)),
                    (actual, expected),
                )
                self.assertEqual(image.size, (16, 16))
                self.assertEqual(image.format, "JPEG")
            self.assertEqual(file_sha256(Path(unit.image_paths[0])), unit.image_sha256[0])
        metadata_path = self.output / "sampling_metadata.json"
        before = metadata_path.read_bytes()
        metadata = json.loads(before)
        self.assertEqual(metadata["frame_count"], 6)
        self.assertEqual(metadata["frame_interval_seconds"], 0.5)
        self.assertEqual(metadata["frames"][0]["image_path"], "candidate_frames/frame_000001.jpg")
        self.assertEqual(metadata["image_format"], "jpeg")
        self.assertEqual(metadata["jpeg_quality"], 6)
        again = sample_video(str(self.video), str(self.output), 0.5)
        self.assertEqual([unit.to_dict() for unit in units], [unit.to_dict() for unit in again])
        self.assertEqual(before, metadata_path.read_bytes())

    def test_rerun_with_larger_interval_removes_stale_samples(self):
        sample_video(str(self.video), str(self.output), 0.5)
        units = sample_video(str(self.video), str(self.output), 2)
        self.assertEqual([unit.timestamp_seconds for unit in units], [0, 2])
        self.assertEqual(len(list((self.output / "candidate_frames").iterdir())), 2)
        self.assertEqual(json.loads((self.output / "sampling_metadata.json").read_text())["frame_count"], 2)

    def test_offset_source_timestamps_still_start_at_first_frame(self):
        video = create_test_video(self.root, offset_seconds=5)
        units = sample_video(str(video), str(self.output))
        self.assertEqual([unit.timestamp_seconds for unit in units], [0, 1, 2])
        for index, unit in enumerate(units):
            with Image.open(unit.image_paths[0]) as image:
                actual = image.getpixel((0, 0))
                expected = (index * 80, 0, 0)
                self.assertTrue(
                    all(abs(value - target) <= 3 for value, target in zip(actual, expected)),
                    (actual, expected),
                )

    def test_unreadable_video_does_not_replace_existing_pool(self):
        units = sample_video(str(self.video), str(self.output))
        metadata = (self.output / "sampling_metadata.json").read_bytes()
        unreadable = self.root / "invalid.mp4"
        unreadable.write_bytes(b"not a video")
        with self.assertRaisesRegex(MontageError, "Cannot read/sample video"):
            sample_video(str(unreadable), str(self.output))
        self.assertEqual(metadata, (self.output / "sampling_metadata.json").read_bytes())
        self.assertEqual(file_sha256(Path(units[0].image_paths[0])), units[0].image_sha256[0])


if __name__ == "__main__":
    unittest.main()
