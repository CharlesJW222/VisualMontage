import unittest
import json
import tempfile
import io
from contextlib import redirect_stderr
from types import SimpleNamespace

from pathlib import Path

from demo import (
    _resolve_dataset,
    _resolve_output_dir,
    _resolve_video,
    _target_output_dirs,
    build_parser,
)


class DemoTests(unittest.TestCase):
    def test_video_sampling_arguments_default_to_three_seconds(self):
        args = build_parser().parse_args(["--video", "/tmp/source.mp4"])
        self.assertEqual(args.video, "/tmp/source.mp4")
        self.assertEqual(args.frame_interval, 3.0)
        args = build_parser().parse_args(["--frame-interval", "0.25"])
        self.assertEqual(args.frame_interval, 0.25)

    def test_manifest_video_file_is_used_when_video_argument_is_omitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "A0FZIwabctw.mp4"
            video.write_bytes(b"video")
            (root / "dataset.json").write_text(
                json.dumps({"video_file": video.name}), encoding="utf-8"
            )
            args = build_parser().parse_args([])
            self.assertEqual(_resolve_video(args, str(root)), str(video.resolve()))

            explicit = build_parser().parse_args(["--video", "/tmp/override.mp4"])
            self.assertEqual(
                _resolve_video(explicit, str(root)), "/tmp/override.mp4"
            )

    def test_target_output_paths_stay_inside_root_and_do_not_collide(self):
        root = Path("/tmp/montage-output")
        ids = ["target_001", "../escape", ".", "..", "candidate_frames", "A", "a", "a/b", "a%2Fb"]
        outputs = _target_output_dirs(root, ids)
        self.assertEqual(outputs[0], root / "target_001")
        self.assertTrue(all(path.parent == root for path in outputs))
        self.assertEqual(len({path.name.casefold() for path in outputs}), len(ids))
        self.assertNotIn(root / "candidate_frames", outputs)
        self.assertEqual(outputs, _target_output_dirs(root, ids))

    def test_default_dataset_and_output_directories_share_content_name(self):
        settings = {
            "dataset": {
                "data_root": "datasets",
                "default_dataset": "x59_simulator_to_first_flight",
            },
            "output": {"directory": "attack_plan"},
        }
        args = SimpleNamespace(dataset=None, output=None)
        dataset = SimpleNamespace(
            root="/tmp/project/datasets/x59_simulator_to_first_flight"
        )
        self.assertEqual(
            _resolve_dataset(args, settings),
            "datasets/x59_simulator_to_first_flight",
        )
        self.assertTrue(
            str(_resolve_output_dir(args, settings, dataset)).endswith(
                "attack_plan/x59_simulator_to_first_flight"
            )
        )

    def test_demo_exposes_four_independent_model_arguments(self):
        args = build_parser().parse_args(
            [
                "--screenwriter-model",
                "screenwriter-vlm",
                "--visual-editor-model",
                "visual-editor-vlm",
                "--caption-editor-model",
                "caption-editor-vlm",
                "--director-model",
                "director-vlm",
            ]
        )
        self.assertEqual(args.screenwriter_model, "screenwriter-vlm")
        self.assertEqual(args.visual_editor_model, "visual-editor-vlm")
        self.assertEqual(args.caption_editor_model, "caption-editor-vlm")
        self.assertEqual(args.director_model, "director-vlm")

    def test_legacy_search_arguments_are_rejected(self):
        for name in ("--visual-editor-beam-width", "--relational-graph-workers"):
            with self.subTest(name=name), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                build_parser().parse_args([name, "2"])

    def test_demo_exposes_optional_victim_validation_arguments(self):
        args = build_parser().parse_args(
            [
                "--enable-victim-validation",
                "--num-victims",
                "5",
                "--num-victim-conclusions",
                "3",
                "--victim-model",
                "victim-vlm",
                "--judge-model",
                "judge-vlm",
                "--target-ground-truth",
                "false",
            ]
        )
        self.assertTrue(args.enable_victim_validation)
        self.assertEqual(args.num_victims, 5)
        self.assertEqual(args.num_victim_conclusions, 3)
        self.assertEqual(args.victim_model, "victim-vlm")
        self.assertEqual(args.judge_model, "judge-vlm")
        self.assertEqual(args.target_ground_truth, "false")

    def test_demo_accepts_heterogeneous_victim_models(self):
        args = build_parser().parse_args(
            [
                "--enable-victim-validation",
                "--victim-models",
                "model-a,model-b,model-c",
            ]
        )
        self.assertIsNone(args.victim_model)
        self.assertEqual(
            args.victim_models, "model-a,model-b,model-c"
        )


if __name__ == "__main__":
    unittest.main()
