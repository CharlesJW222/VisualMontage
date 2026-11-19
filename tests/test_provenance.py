import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from generate_montage_plan import save_montage_plan
from src.production.caption_editor import CaptionSequence
from src.production.director import DirectorDecision
from src.production.screenwriter import ScreenplayOutput
from src.production.visual_editor import VisualSequence
from src.utils.data_loader import ImageUnit, MontageDataset, ProductionSettings, Target
from src.utils.provenance import (
    build_attack_plan_provenance,
    write_attack_plan_provenance,
)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        image_dir = self.root / "images"
        image_dir.mkdir()
        self.image_one = image_dir / "frame_001.png"
        self.image_two = image_dir / "frame_002.png"
        self.image_one.write_bytes(b"first-image")
        self.image_two.write_bytes(b"second-image")
        self.dataset = MontageDataset(
            dataset_id="provenance_dataset",
            root=str(self.root),
            target=Target("target", "A test target."),
            image_units=[
                ImageUnit("frame_001", [str(self.image_one)], ["hash-one"]),
                ImageUnit("frame_002", [str(self.image_two)], ["hash-two"]),
            ],
            settings=ProductionSettings(
                screenwriter_rounds=1,
                visual_editor_rounds=1,
                caption_editor_rounds=1,
                output_frames=2,
                tau=0.7,
                epsilon=0.05,
                search_alpha=0.1,
                search_beta=0.05,
            ),
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def decision():
        return DirectorDecision(
            action="ACCEPT",
            local_compliance=True,
            target_belief=0.9,
            singleton_leakage=0.02,
            violations=[],
            critique="Accepted for test.",
            agent_id="director",
            model="mock-model",
            auditor_beliefs=[0.9],
            modality_loads={"v": 0.02, "t": 0.0, "vt": 0.0},
            lc_components={
                "authenticity": True,
                "caption_faithfulness": True,
                "local_visual_support": True,
            },
        )

    def make_plan(self):
        screenplay = ScreenplayOutput(
            narrative="A setup precedes an outcome.",
            target_relations=[
                {
                    "source_role": "setup",
                    "relation": "leads to",
                    "destination_role": "outcome",
                }
            ],
            event_roles=[
                {
                    "role_id": "setup",
                    "visual_requirement": "A setup is visible.",
                    "narrative_function": "Open.",
                },
                {
                    "role_id": "outcome",
                    "visual_requirement": "An outcome is visible.",
                    "narrative_function": "Close.",
                },
            ],
            composition_strategy="Order the two frames.",
        )
        visual = VisualSequence(
            ordered_unit_ids=["frame_001", "frame_002"],
            sequencing_strategy="Use the strongest directed transition.",
        )
        captions = CaptionSequence(
            units=[
                {"unit_id": "frame_001", "modality": "v", "caption": ""},
                {
                    "unit_id": "frame_002",
                    "modality": "vt",
                    "caption": "A locally visible outcome.",
                },
            ],
            caption_strategy="Use one grounded caption.",
        )
        decision = self.decision()
        return SimpleNamespace(
            dataset=self.dataset,
            screenplay=screenplay,
            screenplay_decision=decision,
            visual_sequence=visual,
            visual_sequence_decision=decision,
            caption_sequence=captions,
            caption_sequence_decision=decision,
            publication_plan=[
                {
                    "position": 1,
                    "unit_id": "frame_001",
                    "modality": "v",
                    "caption": None,
                    "account_id": "account_a",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                },
                {
                    "position": 2,
                    "unit_id": "frame_002",
                    "modality": "vt",
                    "caption": "A locally visible outcome.",
                    "account_id": "account_b",
                    "timestamp": "2026-01-01T00:10:00+00:00",
                },
            ],
            agent_models={"director": {"agent_id": "director", "model": "mock-model"}},
            runtime_diagnostics={"elapsed_seconds": 1.0},
            debate_history=[
                {
                    "phase": "visual_editor",
                    "round": 1,
                    "draft": visual.to_dict(),
                    "director": decision.to_dict(),
                }
            ],
        )

    def test_writes_readable_and_structured_final_sequence_trace(self):
        output = self.root / "attack_plan"
        sequence = output / "sequence" / "captioned_image_sequence.json"
        sequence.parent.mkdir(parents=True)
        sequence.write_text(
            json.dumps(
                {
                    "items": [
                        {
                            "unit_id": "frame_001",
                            "feed_item_id": "montage_0001",
                            "image_path": "victim_images/frame_0001.png",
                            "rendered_image_sha256": "rendered-one",
                            "image_withheld": False,
                            "subtitle_burned": False,
                        },
                        {
                            "unit_id": "frame_002",
                            "feed_item_id": "montage_0002",
                            "image_path": "victim_images/frame_0002.png",
                            "rendered_image_sha256": "rendered-two",
                            "image_withheld": False,
                            "subtitle_burned": True,
                            "subtitle_font_size": 18,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        paths = write_attack_plan_provenance(
            self.make_plan(),
            str(output),
            materialized={"captioned_image_sequence": str(sequence)},
            victim_validation={"performed": False, "status": "not_run"},
        )
        record = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        self.assertEqual(record["schema_version"], "visual_montage.attack_plan_provenance.v1")
        self.assertNotIn("continuity_penalty", record["selection_score"])
        self.assertNotIn("continuity_penalty", record["final_sequence"][0]["visual_selection"])
        self.assertEqual(record["final_sequence"][1]["unit_id"], "frame_002")
        self.assertIsNone(
            record["final_sequence"][1]["relational_beliefs"]["edge_from_previous"],
        )
        self.assertEqual(
            record["final_sequence"][1]["final_modality_and_caption"]["modality"],
            "vt",
        )
        self.assertEqual(
            record["final_sequence"][1]["materialized_output"][0]["subtitle_burned"],
            True,
        )
        report = Path(paths["markdown"]).read_text(encoding="utf-8")
        self.assertIn("# Attack Plan Provenance Record", report)
        self.assertIn("Per-item provenance of the final sequence", report)
        self.assertIn("hash-two", report)

    def test_plan_saver_emits_initial_provenance_without_materialization(self):
        plan = self.make_plan()
        plan.to_dict = lambda: {"schema_version": "test.plan"}
        plan.runtime_diagnostics = {}
        output = self.root / "saved_plan"
        save_montage_plan(plan, str(output))
        self.assertTrue(
            (output / "provenance" / "attack_plan_provenance.json").is_file()
        )
        self.assertTrue(
            (output / "provenance" / "attack_plan_provenance.md").is_file()
        )

    def test_no_graph_export_retains_gates_without_fabricating_scores(self):
        class UnusedSearchSetting(float):
            def __float__(self):
                raise AssertionError("Inactive search settings must not be evaluated")

        plan = self.make_plan()
        plan.dataset.settings.search_alpha = UnusedSearchSetting(0.9)
        plan.dataset.settings.search_beta = UnusedSearchSetting(0.8)
        record = build_attack_plan_provenance(plan, str(self.root / "active"))
        score = record["selection_score"]
        self.assertNotIn("continuity_penalty", score)
        self.assertEqual(score["status"], "not_used")
        for key in (
            "formula",
            "selected_adjacent_omega_sum",
            "alpha",
            "beta",
            "alpha_times_continuity_penalty",
            "beta_times_selected_frame_count",
            "computed_search_score",
        ):
            self.assertIsNone(score[key], key)
        self.assertEqual(score["selected_frame_count"], 2)
        self.assertEqual(score["selected_edges"], [])
        for item in record["final_sequence"]:
            self.assertEqual(
                item["relational_beliefs"],
                {
                    "status": "not_used",
                    "singleton_belief": None,
                    "edge_from_previous": None,
                    "edge_to_next": None,
                },
            )
        self.assertEqual(record["stage_gates"]["visual_editor"]["action"], "ACCEPT")
        self.assertIsNone(record["candidate_trace"][0]["search_score"])

    def test_no_graph_video_trace_preserves_source_and_zero_timestamp(self):
        plan = self.make_plan()
        source_video = str(self.root / "source.mp4")
        for timestamp, unit in enumerate(plan.dataset.image_units):
            unit.timestamp_seconds = float(timestamp)
            unit.source_video = source_video
        paths = write_attack_plan_provenance(plan, str(self.root / "video"))
        record = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        first, second = record["final_sequence"]
        self.assertEqual(first["unit_id"], "frame_001")
        self.assertEqual(first["timestamp_seconds"], 0.0)
        self.assertEqual(second["timestamp_seconds"], 1.0)
        self.assertEqual(first["source_video"], source_video)
        self.assertEqual(
            first["source_images"][0]["absolute_path"], str(self.image_one.resolve())
        )
        report = Path(paths["markdown"]).read_text(encoding="utf-8")
        self.assertIn("not_used", report)
        self.assertIn(source_video, report)
        self.assertIn("Sampling time (seconds):** 0.0", report)
        self.assertNotIn("sum(selected adjacent omega)", report)


if __name__ == "__main__":
    unittest.main()
