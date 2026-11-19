import hashlib
import io
import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import demo
from demo import build_parser
from src.utils.data_loader import MontageError, load_dataset
from test_video_sampler import create_test_video


class ScriptedClient:
    def __init__(self, role):
        self.model_name = role
        self.calls = []

    def generate_with_images(self, prompt, image_paths, system_prompt=None, image_descriptors=None):
        self.calls.append((prompt, list(image_paths)))
        ids = [descriptor.split()[0].split("=", 1)[1] for descriptor in image_descriptors]
        if self.model_name == "screenwriter":
            result = {
                "narrative": "A preparation precedes an outcome.",
                "target_relations": [{"source_role": "setup", "relation": "precedes", "destination_role": "outcome"}],
                "event_roles": [
                    {"role_id": "setup", "visual_requirement": "A dark surface.", "narrative_function": "Establish a setting."},
                    {"role_id": "outcome", "visual_requirement": "A brighter surface.", "narrative_function": "Show a change."},
                ],
                "composition_strategy": "Juxtapose two visible states.",
            }
        elif self.model_name == "visual_editor":
            result = {"ordered_unit_ids": [ids[-1], ids[0]], "sequencing_strategy": "Show the outcome before its context.", "continuity_penalty": 0.0}
        elif self.model_name == "caption_editor":
            result = {"units": [{"unit_id": unit_id, "modality": "v", "caption": ""} for unit_id in ids], "caption_strategy": "Preserve the approved visual order."}
        else:
            result = {
                "action": "ACCEPT", "locally_truthful": True, "target_belief": 0.9,
                "auditor_beliefs": [0.9], "singleton_leakage": 0.0,
                "modality_loads": {"v": 0.0, "t": 0.0, "vt": 0.0},
                "lc_components": {"authenticity": True, "caption_faithfulness": True, "local_visual_support": True},
                "revise_owner": "none", "spurious_edges": [], "violations": [],
                "critique": "The sequence satisfies the gate criteria.",
            }
        return json.dumps(result)

    def generate_multimodal(self, content, system_prompt=None):
        self.calls.append(content)
        return json.dumps({
            "conclusions": [{"rank": 1, "conclusion": "Two visible states differ.", "confidence": 0.9, "verdict": "true", "should_publish": True, "rationale": "The visible surfaces differ.", "publication_rationale": "The conclusion is locally grounded."}],
            "final_verdict": {"verdict": "false", "confidence": 0.9, "rationale": "The visible states alone do not establish the claim."},
        })


class MultiTargetDatasetTests(unittest.TestCase):
    def test_loads_targets_array_and_selects_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "images" / "frame_001.png"
            image.parent.mkdir()
            image.write_bytes(b"test-image")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            (root / "image_sequence.json").write_text(
                json.dumps(
                    {
                        "image_sequence": [
                            {
                                "id": "frame_001",
                                "image_path": "images/frame_001.png",
                                "image_sha256": digest,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "dataset.json").write_text(
                json.dumps(
                    {
                        "schema_version": "visual_montage.target_image_sequence.v1",
                        "dataset_id": "multi_target_test",
                        "ground_truth": False,
                        "targets": [
                            {"id": "target_a", "label": "First claim."},
                            {"id": "target_b", "label": "Second claim."},
                        ],
                        "image_sequence_file": "image_sequence.json",
                        "settings": {"output_frames": 1},
                    }
                ),
                encoding="utf-8",
            )

            dataset = load_dataset(str(root))
            self.assertEqual(dataset.target_ids, ["target_a", "target_b"])
            self.assertEqual(dataset.target.target_id, "target_a")
            self.assertEqual(
                dataset.select_target("target_b").target.label,
                "Second claim.",
            )

    def test_demo_exposes_target_id(self):
        args = build_parser().parse_args(["--target-id", "target_b"])
        self.assertEqual(args.target_id, "target_b")

    def test_video_loader_accepts_targets_settings_without_image_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "dataset.json").write_text(json.dumps({
                "schema_version": "visual_montage.target_image_sequence.v1",
                "dataset_id": "video_test", "target": {"id": "claim", "label": "A manually specified claim."},
                "settings": {"output_frames": 6},
            }))
            dataset = load_dataset(str(root), load_images=False)
            self.assertEqual(dataset.image_units, [])
            self.assertEqual(dataset.target_ids, ["claim"])
            self.assertEqual(dataset.settings.frame_selection_count, 6)
            with self.assertRaisesRegex(MontageError, "Cannot read JSON"):
                load_dataset(str(root))


class DemoPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "dataset"
        self.dataset_root.mkdir()
        self.output = self.root / "results"
        self.clients = []
        self.victims = []

    def write_dataset(self, target_ids=("target_a", "target_b"), *, images=False, output_frames=2):
        manifest = {
            "schema_version": "visual_montage.target_image_sequence.v1",
            "dataset_id": "integration_test",
            "targets": [{"id": target_id, "label": "Manually declared claim %s." % target_id} for target_id in target_ids],
            "settings": {"output_frames": output_frames, "screenwriter_rounds": 1, "visual_editor_rounds": 1, "caption_editor_rounds": 1},
        }
        (self.dataset_root / "dataset.json").write_text(json.dumps(manifest))
        if images:
            sequence = []
            for index in range(1, 4):
                name = "frame_%03d" % index
                image = self.dataset_root / (name + ".png")
                Image.new("RGB", (16, 16), (index * 50, 0, 0)).save(image)
                sequence.append({"id": name, "image_path": image.name, "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
            (self.dataset_root / "image_sequence.json").write_text(json.dumps({"image_sequence": sequence}))

    def run_demo(self, extra=()):
        settings = {"llm": {"roles": {role: {"model": role} for role in ("screenwriter", "visual_editor", "caption_editor", "director")}}}

        def client_factory(values, overrides):
            client = ScriptedClient(values["model"])
            self.clients.append(client)
            return client

        def victim_factory(*args, **kwargs):
            client = ScriptedClient("victim")
            self.victims.append(client)
            return client

        arguments = [
            "demo.py", "--dataset", str(self.dataset_root), "--output", str(self.output),
            "--quiet", "--frame-interval", "1", *extra,
        ]
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", arguments))
            agents = json.loads(
                (Path(__file__).resolve().parents[1] / "configs/agents.json").read_text()
            )
            stack.enter_context(patch("demo.load_configs", return_value=(settings, agents)))
            stack.enter_context(patch("tools.create_private_client", side_effect=client_factory))
            stack.enter_context(patch("demo.build_validation_client", side_effect=victim_factory))
            sampler = stack.enter_context(patch("demo.sample_video", wraps=demo.sample_video))
            generator = stack.enter_context(patch("demo.generate_montage_plan", wraps=demo.generate_montage_plan))
            stack.enter_context(redirect_stdout(io.StringIO()))
            result = demo.main()
        self.assertEqual(result, 0)
        return sampler, generator

    def test_image_only_single_target_keeps_existing_output_layout(self):
        self.write_dataset(target_ids=("only_target",), images=True)
        sampler, generator = self.run_demo()
        sampler.assert_not_called()
        self.assertEqual(generator.call_count, 1)
        plan = json.loads((self.output / "montage_plan.json").read_text())
        self.assertEqual(plan["input"]["target"]["id"], "only_target")
        self.assertNotIn("source_video", plan["input"]["image_sequence"][0])
        self.assertFalse((self.output / "only_target").exists())

    def test_image_only_multiple_targets_run_sequentially(self):
        self.write_dataset(images=True)
        sampler, generator = self.run_demo()
        sampler.assert_not_called()
        self.assertEqual([call.kwargs["dataset"].target.target_id for call in generator.call_args_list], ["target_a", "target_b"])
        self.assertIs(generator.call_args_list[0].kwargs["dataset"].image_units, generator.call_args_list[1].kwargs["dataset"].image_units)
        self.assertTrue((self.output / "target_a" / "montage_plan.json").is_file())
        self.assertTrue((self.output / "target_b" / "montage_plan.json").is_file())

    def test_image_output_override_is_checked_after_manifest_defaults(self):
        self.write_dataset(target_ids=("only_target",), images=True, output_frames=6)
        self.run_demo(["--output-frames", "2"])
        plan = json.loads((self.output / "montage_plan.json").read_text())
        self.assertEqual(plan["settings"]["output_frames"], 2)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required for video integration")
    def test_video_all_targets_share_one_pool_and_each_run_victim_validation(self):
        self.write_dataset()
        video = create_test_video(self.root)
        sampler, generator = self.run_demo(["--video", str(video), "--enable-victim-validation", "--num-victims", "1", "--skip-ai-judge"])
        sampler.assert_called_once()
        self.assertEqual(generator.call_count, 2)
        first, second = [call.kwargs["dataset"] for call in generator.call_args_list]
        self.assertIs(first.image_units, second.image_units)
        self.assertEqual(len(first.image_units), 3)
        self.assertEqual(len(self.victims), 2)
        self.assertTrue(all(len(client.calls) == 1 for client in self.victims))
        self.assertEqual(len(list(self.output.rglob("sampling_metadata.json"))), 1)
        self.assertEqual(len(list(self.output.rglob("candidate_frames"))), 1)
        self.assertEqual(len(list((self.output / "candidate_frames").glob("*.jpg"))), 3)
        for target_id in ("target_a", "target_b"):
            directory = self.output / target_id
            plan = json.loads((directory / "montage_plan.json").read_text())
            self.assertEqual(plan["input"]["target"]["id"], target_id)
            self.assertIsNone(plan["relational_emergence_graph"])
            self.assertEqual(plan["input"]["image_sequence"][0]["timestamp_seconds"], 0)
            self.assertEqual(plan["input"]["image_sequence"][0]["source_video"], str(video.resolve()))
            self.assertTrue(plan["victim_validation"]["performed"])
            validation = json.loads((directory / "victim_validation.json").read_text())
            self.assertEqual(validation["metadata"]["target_conclusion"], "Manually declared claim %s." % target_id)
            self.assertEqual(validation["victim_verdicts"][0]["verdict"], "false")
            self.assertEqual(len(validation["victim_feed"]["items"]), 2)
            self.assertTrue((directory / "run_progress.json").is_file())
            self.assertEqual(len(list(directory.rglob("attack_plan_provenance.json"))), 1)
            self.assertFalse((directory / "candidate_frames").exists())
        for client in self.clients:
            if client.model_name in ("screenwriter", "visual_editor"):
                self.assertEqual(len(client.calls[0][1]), 3)
            elif client.model_name == "caption_editor":
                self.assertEqual(len(client.calls[0][1]), 2)
            elif client.model_name == "director":
                self.assertEqual(len(client.calls), 3)
            if client.model_name in ("screenwriter", "visual_editor", "caption_editor"):
                self.assertTrue(client.calls[0][1][0].endswith(".jpg"))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required for video integration")
    def test_video_explicit_target_runs_only_selected_target_in_existing_layout(self):
        self.write_dataset(output_frames=6)
        video = create_test_video(self.root)
        sampler, generator = self.run_demo(["--video", str(video), "--target-id", "target_b", "--output-frames", "2"])
        sampler.assert_called_once()
        self.assertEqual(generator.call_count, 1)
        self.assertEqual(generator.call_args.kwargs["dataset"].target.target_id, "target_b")
        self.assertTrue((self.output / "montage_plan.json").is_file())
        self.assertFalse((self.output / "target_a").exists())
        self.assertFalse((self.output / "target_b").exists())

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required for video integration")
    def test_video_single_target_runs_normally(self):
        self.write_dataset(target_ids=("only_target",))
        video = create_test_video(self.root)
        sampler, generator = self.run_demo(["--video", str(video)])
        sampler.assert_called_once()
        self.assertEqual(generator.call_count, 1)
        self.assertTrue((self.output / "montage_plan.json").is_file())

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required for video integration")
    def test_video_pool_below_configured_count_fails_before_any_model(self):
        self.write_dataset(output_frames=6)
        video = create_test_video(self.root)
        with self.assertRaisesRegex(MontageError, "frame selection count exceeds"):
            self.run_demo(["--video", str(video)])
        self.assertEqual(self.clients, [])


if __name__ == "__main__":
    unittest.main()
