import json
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import build_production_team, load_configs
from src.production.screenwriter import ScreenplayOutput
from src.production.visual_editor import (
    VISUAL_VALIDATION_RETRY_LIMIT,
    VisualEditor,
)
from src.utils.data_loader import (
    ImageUnit,
    MontageDataset,
    MontageError,
    ProductionSettings,
    Target,
)


class RecordingAgent:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_json(self, prompt, image_units):
        self.calls.append((prompt, image_units))
        return next(self.responses)


class VisualEditorTests(unittest.TestCase):
    def setUp(self):
        self.dataset = MontageDataset(
            dataset_id="test",
            root="/unused",
            target=Target("target_a", "The preparation led to the outcome."),
            image_units=[
                ImageUnit("frame_%06d" % index, [], []) for index in range(1, 5)
            ],
            settings=ProductionSettings(output_frames=2),
        )
        self.screenplay = ScreenplayOutput(
            narrative="Preparation precedes an outcome.",
            target_relations=[
                {
                    "source_role": "preparation",
                    "relation": "appears to enable",
                    "destination_role": "outcome",
                }
            ],
            event_roles=[
                {
                    "role_id": role,
                    "visual_requirement": "A visible event.",
                    "narrative_function": "Establish the event.",
                }
                for role in ("preparation", "outcome")
            ],
            composition_strategy="Juxtapose the events in presentation order.",
        )
        self.valid = {
            "ordered_unit_ids": ["frame_000004", "frame_000001"],
            "sequencing_strategy": "Use complementary frames in reverse source order.",
            "continuity_penalty": 0.1,
        }

    def test_one_proposal_receives_complete_pool_and_configured_count(self):
        agent = RecordingAgent([self.valid])
        sequence = VisualEditor(agent).create_sequence(self.dataset, self.screenplay)
        self.assertEqual(
            sequence.to_dict(),
            {key: value for key, value in self.valid.items() if key != "continuity_penalty"},
        )
        self.assertNotIn("continuity_penalty", sequence.to_dict())
        self.assertEqual(len(agent.calls), 1)
        prompt, attached_units = agent.calls[0]
        self.assertNotIn("continuity_penalty", prompt)
        self.assertIs(attached_units, self.dataset.image_units)
        for unit in self.dataset.image_units:
            self.assertIn(unit.unit_id, prompt)
        self.assertIn("Select exactly 2 distinct supplied frame IDs", prompt)
        self.assertIn(self.dataset.target.label, prompt)
        self.assertIn(self.screenplay.narrative, prompt)
        for legacy_field in ("omega", "walk_score", "retained_candidates"):
            self.assertNotIn(legacy_field, prompt)

    def test_revision_receives_previous_proposal_and_director_feedback(self):
        previous = dict(self.valid, ordered_unit_ids=["frame_000001", "frame_000002"])
        critique = "Replace redundant evidence to establish the intermediate event."
        agent = RecordingAgent([self.valid])
        VisualEditor(agent).create_sequence(
            self.dataset,
            self.screenplay,
            round_number=2,
            previous=previous,
            director_critique=critique,
        )
        prompt = agent.calls[0][0]
        self.assertIn("Visual Editor round 2", prompt)
        self.assertIn(json.dumps(previous, ensure_ascii=False), prompt)
        self.assertIn(critique, prompt)

    def test_invalid_frame_selections_are_rejected(self):
        cases = (
            ("frame_000001", "must be an array"),
            (["frame_000001", "frame_000001"], "must be distinct"),
            (["frame_000001", "frame_999999"], "unknown frame IDs"),
            (["frame_000001"], "must select exactly 2"),
            (["frame_000001", "frame_000002", "frame_000003"], "must select exactly 2"),
        )
        for ordered_ids, error in cases:
            with self.subTest(ordered_ids=ordered_ids):
                with self.assertRaisesRegex(MontageError, error):
                    VisualEditor._parse_response(
                        dict(self.valid, ordered_unit_ids=ordered_ids), self.dataset
                    )

    def test_structural_retry_uses_same_full_pool_and_error_feedback(self):
        invalid = dict(self.valid, ordered_unit_ids=["frame_000001", "frame_000001"])
        agent = RecordingAgent([invalid, self.valid])
        result = VisualEditor(agent).create_sequence(self.dataset, self.screenplay)
        self.assertEqual(
            result.to_dict(),
            {key: value for key, value in self.valid.items() if key != "continuity_penalty"},
        )
        self.assertEqual(len(agent.calls), 2)
        self.assertIn("frame IDs must be distinct", agent.calls[1][0])
        self.assertIs(agent.calls[1][1], self.dataset.image_units)

    def test_structural_retry_limit_remains_bounded(self):
        agent = RecordingAgent([{}] * (VISUAL_VALIDATION_RETRY_LIMIT + 1))
        with self.assertRaisesRegex(MontageError, "must be an array"):
            VisualEditor(agent).create_sequence(self.dataset, self.screenplay)
        self.assertEqual(len(agent.calls), VISUAL_VALIDATION_RETRY_LIMIT + 1)

    def test_insufficient_pool_fails_before_model_call(self):
        self.dataset.settings.output_frames = 5
        agent = RecordingAgent([])
        with self.assertRaisesRegex(MontageError, "fewer frames"):
            VisualEditor(agent).create_sequence(self.dataset, self.screenplay)
        self.assertEqual(agent.calls, [])

    def test_production_team_uses_configured_prompts(self):
        path = Path(__file__).resolve().parents[1] / "configs"
        settings, config = load_configs(str(path))
        with patch("tools._build_client"):
            team = build_production_team(
                settings, config, self.dataset.settings, cache_path=None
            )
        agents = {
            "screenwriter": team.screenwriter.prompt_agent,
            "visual_editor": team.visual_editor.prompt_agent,
            "caption_editor": team.caption_editor.prompt_agent,
            "director": team.prompt_agent,
        }
        for role, agent in agents.items():
            with self.subTest(role=role):
                self.assertEqual(
                    agent.system_prompt,
                    config["production_team"][role]["system_prompt"],
                )


if __name__ == "__main__":
    unittest.main()
