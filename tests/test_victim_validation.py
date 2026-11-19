import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.utils.data_loader import MontageError
from victim_validation import (
    build_parser as build_victim_parser,
    build_multimodal_victim_feed,
    build_victim_request_content,
    calculate_ddr_metrics,
    calculate_victim_metrics,
    majority_vote_ruling,
    resolve_victim_configuration,
    run_victim_validation,
)


class MockMultimodalClient:
    def __init__(self, responses, model_name="mock-victim-vlm"):
        self.responses = list(responses)
        self.model_name = model_name
        self.calls = []

    def generate_multimodal(self, content, system_prompt=None):
        self.calls.append(
            {"content": list(content), "system_prompt": system_prompt}
        )
        return json.dumps(self.responses.pop(0), ensure_ascii=False)


class VictimValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        victim_images = self.root / "sequence" / "victim_images"
        victim_images.mkdir(parents=True)
        for index in (1, 3):
            Image.new("RGB", (80, 60), (index * 60, 40, 100)).save(
                str(victim_images / ("frame_%04d.png" % index)),
                format="PNG",
            )
        hidden = self.root / "hidden.png"
        Image.new("RGB", (80, 60), (10, 20, 30)).save(
            str(hidden), format="PNG"
        )
        self.hidden_path = str(hidden)
        self.sequence_path = (
            self.root / "sequence" / "captioned_image_sequence.json"
        )
        self.sequence_path.write_text(
            json.dumps(
                {
                    "target": {"label": "Private target metadata"},
                    "items": [
                        {
                            "sequence_position": 1,
                            "display_timestamp": "2026-01-01T10:00:00Z",
                            "publisher_id": "account_a",
                            "image_path": str(
                                victim_images / "frame_0001.png"
                            ),
                            "caption": None,
                            "image_withheld": False,
                            "modality": "v",
                            "source_image_path": self.hidden_path,
                            "director_critique": "private",
                        },
                        {
                            "sequence_position": 2,
                            "display_timestamp": "2026-01-01T10:10:00Z",
                            "publisher_id": "account_b",
                            "image_path": None,
                            "caption": "A visible text-only caption.",
                            "image_withheld": True,
                            "modality": "t",
                            "source_image_path": self.hidden_path,
                        },
                        {
                            "sequence_position": 3,
                            "display_timestamp": "2026-01-01T10:20:00Z",
                            "publisher_id": "account_a",
                            "image_path": str(
                                victim_images / "frame_0003.png"
                            ),
                            "caption": "A visible image caption.",
                            "image_withheld": False,
                            "modality": "vt",
                            "source_image_path": self.hidden_path,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def victim_response(verdict, confidence):
        return {
            "conclusions": [
                {
                    "rank": 1,
                    "conclusion": "A core visible conclusion.",
                    "confidence": confidence,
                    "verdict": "true",
                    "should_publish": True,
                    "rationale": "Visible evidence supports it.",
                    "publication_rationale": (
                        "It is sufficiently supported."
                    ),
                }
            ],
            "final_verdict": {
                "target_conclusion": "Specified claim",
                "verdict": verdict,
                "confidence": confidence,
                "rationale": (
                    "Ordered visible evidence supports this verdict."
                ),
            },
        }

    def test_feed_exposes_only_visible_content_in_final_order(self):
        feed = build_multimodal_victim_feed(self.sequence_path)
        self.assertEqual(
            [item["sequence_position"] for item in feed["items"]],
            [1, 2, 3],
        )
        self.assertIsNotNone(feed["items"][0]["image_path"])
        self.assertIsNone(feed["items"][0]["text"])
        self.assertIsNone(feed["items"][1]["image_path"])
        self.assertEqual(
            feed["items"][1]["text"], "A visible text-only caption."
        )
        serialized = json.dumps(feed)
        self.assertNotIn("modality", serialized)
        self.assertNotIn("director_critique", serialized)
        self.assertNotIn(self.hidden_path, serialized)
        self.assertNotIn("Private target metadata", serialized)

    def test_request_interleaves_images_and_text_without_modality_labels(self):
        feed = build_multimodal_victim_feed(self.sequence_path)
        content = build_victim_request_content(
            feed, "Specified claim", num_conclusions=1
        )
        block_types = [block["type"] for block in content]
        self.assertEqual(
            block_types,
            [
                "text",
                "text",
                "image_path",
                "text",
                "text",
                "text",
                "image_path",
                "text",
                "text",
            ],
        )
        visible_text = "\n".join(
            block.get("text", "") for block in content
        )
        self.assertNotIn(" v ", visible_text)
        self.assertNotIn(" t ", visible_text)
        self.assertNotIn("vt", visible_text)
        self.assertNotIn(self.hidden_path, json.dumps(content))

    def test_full_validation_matches_text_metrics_and_downstream_shapes(self):
        victim_client = MockMultimodalClient(
            [
                self.victim_response("true", 0.9),
                self.victim_response("false", 0.7),
            ]
        )
        judge_client = MockMultimodalClient(
            [
                {
                    "decision": "accept",
                    "confidence": 0.85,
                    "rationale": (
                        "The visible feed and reports support acceptance."
                    ),
                }
            ]
        )
        result = run_victim_validation(
            sequence=self.sequence_path,
            target_conclusion="Specified claim",
            ground_truth="false",
            num_victims=2,
            victim_client=victim_client,
            num_conclusions=1,
            high_confidence_threshold=0.8,
            judge_client=judge_client,
            verbose=False,
        )
        self.assertEqual(len(victim_client.calls), 2)
        self.assertEqual(
            result["metadata"]["victim_model_mode"],
            "single_model_repeated",
        )
        self.assertEqual(
            result["metadata"]["victim_models"],
            ["mock-victim-vlm", "mock-victim-vlm"],
        )
        self.assertEqual(result["analysis"]["ASR"], 50.0)
        self.assertEqual(result["analysis"]["average_confidence"], 0.8)
        self.assertEqual(result["analysis"]["HC_ASR"], 50.0)
        victim = result["victim_verdicts"][0]
        required_keys = {
            "victim_id",
            "model",
            "conclusions",
            "verdict",
            "confidence",
            "reasoning",
            "full_response",
        }
        self.assertTrue(required_keys.issubset(victim))
        self.assertEqual(
            result["adjudication"]["majority_vote"]["decision"],
            "reject",
        )
        self.assertEqual(
            result["adjudication"]["ai_judge"]["decision"],
            "accept",
        )
        self.assertEqual(
            result["adjudication"]["metrics"],
            {"majority_vote_ddr": 0, "ai_judge_ddr": 1},
        )

    def test_heterogeneous_victim_clients_are_mapped_one_to_one(self):
        clients = [
            MockMultimodalClient(
                [self.victim_response("true", 0.8)], "model-a"
            ),
            MockMultimodalClient(
                [self.victim_response("false", 0.9)], "model-b"
            ),
            MockMultimodalClient(
                [self.victim_response("true", 0.7)], "model-c"
            ),
        ]
        result = run_victim_validation(
            sequence=self.sequence_path,
            target_conclusion="Specified claim",
            ground_truth="false",
            num_victims=3,
            victim_client=clients,
            num_conclusions=1,
            judge_client=None,
            verbose=False,
        )
        self.assertEqual([len(client.calls) for client in clients], [1, 1, 1])
        self.assertEqual(
            [item["model"] for item in result["victim_verdicts"]],
            ["model-a", "model-b", "model-c"],
        )
        self.assertEqual(
            result["metadata"]["victim_models"],
            ["model-a", "model-b", "model-c"],
        )
        self.assertEqual(
            result["metadata"]["victim_model_mode"], "model_list"
        )

    def test_victim_model_cli_supports_both_modes(self):
        single = build_victim_parser().parse_args(
            [
                "--sequence",
                "sequence.json",
                "--victim-model",
                "model-a",
                "--num-victims",
                "5",
            ]
        )
        self.assertEqual(single.victim_model, "model-a")
        self.assertIsNone(single.victim_models)

        heterogeneous = build_victim_parser().parse_args(
            [
                "--sequence",
                "sequence.json",
                "--victim-models",
                "model-a,model-b,model-c",
            ]
        )
        num_victims, models = resolve_victim_configuration(
            heterogeneous.victim_models,
            heterogeneous.num_victims,
            5,
        )
        self.assertEqual(num_victims, 3)
        self.assertEqual(models, ["model-a", "model-b", "model-c"])
        with self.assertRaisesRegex(MontageError, "must equal"):
            resolve_victim_configuration(
                "model-a,model-b",
                requested_num_victims=3,
                default_num_victims=5,
            )

    def test_metric_and_ddr_helpers_keep_text_semantics(self):
        verdicts = [
            {"verdict": "true", "confidence": 0.8},
            {"verdict": "true", "confidence": 0.79},
            {"verdict": "error", "confidence": 0.0},
        ]
        metrics = calculate_victim_metrics(verdicts, "false", 0.8)
        self.assertAlmostEqual(metrics["ASR"], 200.0 / 3.0, places=4)
        self.assertAlmostEqual(metrics["HC_ASR"], 100.0 / 3.0, places=4)
        majority = majority_vote_ruling(verdicts)
        judge = {"decision": "reject"}
        self.assertEqual(majority["decision"], "accept")
        self.assertEqual(
            calculate_ddr_metrics(majority, judge, "false"),
            {"majority_vote_ddr": 1, "ai_judge_ddr": 0},
        )


if __name__ == "__main__":
    unittest.main()
