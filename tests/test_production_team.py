import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.environment import ImagePublisher
from src.production import (
    CaptionEditor,
    Director,
    Screenwriter,
    VisualEditor,
    VisualSequence,
)
from src.utils.agent_runtime import AgentCache, PromptAgent
from src.utils.data_loader import MontageError, ProductionSettings, load_dataset


class MockVLMClient:
    def __init__(self, model_name, responder):
        self.model_name = model_name
        self.responder = responder
        self.calls = []

    def generate_with_images(
        self, prompt, image_paths, system_prompt=None, image_descriptors=None
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "image_paths": list(image_paths),
                "system_prompt": system_prompt,
                "image_descriptors": list(image_descriptors or []),
            }
        )
        return json.dumps(self.responder(prompt), ensure_ascii=False)

    def performance_stats(self):
        return {"mock": True, "calls": len(self.calls)}


def create_dataset(root):
    images = root / "images"
    images.mkdir()
    sequence = []
    for index in range(1, 4):
        relative = "images/frame_%03d.png" % index
        Image.new(
            "RGB",
            (160, 90),
            (index * 50, 40 + index * 20, 120),
        ).save(str(root / relative), format="PNG")
        data = (root / relative).read_bytes()
        sequence.append(
            {
                "id": "frame_%03d" % index,
                "image_path": relative,
                "image_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "schema_version": "visual_montage.target_image_sequence.v1",
        "dataset_id": "test_dataset",
        "target": {"id": "target", "label": "unsupported target"},
        "image_sequence_file": "image_sequence.json",
        "settings": {
            "screenwriter_rounds": 1,
            "visual_editor_rounds": 1,
            "caption_editor_rounds": 1,
            "output_frames": 2,
            "tau": 0.8,
            "epsilon": 0.05,
        },
    }
    (root / "dataset.json").write_text(json.dumps(manifest), "utf-8")
    (root / "image_sequence.json").write_text(
        json.dumps({"image_sequence": sequence}), "utf-8"
    )
class ProductionTeamTests(unittest.TestCase):
    def test_old_graph_module_is_unavailable(self):
        self.assertIsNone(importlib.util.find_spec("src.production.relational_graph"))
        self.assertIsNone(importlib.util.find_spec("src.production._emergence_scoring"))

    def test_removed_search_settings_are_rejected(self):
        for name in ("visual_editor_beam_width", "relational_graph_workers"):
            with self.subTest(name=name), self.assertRaisesRegex(MontageError, "Unsupported settings"):
                ProductionSettings.from_dict({name: 2})

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        create_dataset(self.root)
        self.dataset = load_dataset(str(self.root))
        self.cache = AgentCache(None)

    def tearDown(self):
        self.temporary.cleanup()

    def test_loader_rejects_semantic_frame_ids(self):
        path = self.root / "image_sequence.json"
        payload = json.loads(path.read_text("utf-8"))
        payload["image_sequence"][0]["id"] = "setup_frame"
        path.write_text(json.dumps(payload), "utf-8")
        with self.assertRaisesRegex(MontageError, "frame_###"):
            load_dataset(str(self.root))

    def test_loader_rejects_grouped_frame_records(self):
        path = self.root / "image_sequence.json"
        payload = json.loads(path.read_text("utf-8"))
        first = payload["image_sequence"][0]
        payload["image_sequence"][0] = {
            "id": first["id"],
            "image_paths": [first["image_path"]],
            "image_sha256": [first["image_sha256"]],
        }
        path.write_text(json.dumps(payload), "utf-8")
        with self.assertRaisesRegex(MontageError, "unsupported keys"):
            load_dataset(str(self.root))

    def make_agent(self, role, model, responder):
        return PromptAgent(
            role,
            MockVLMClient(model, responder),
            "%s system prompt" % role,
            self.cache,
        )

    @staticmethod
    def accepted(prompt):
        return {
            "action": "ACCEPT",
            "locally_truthful": True,
            "target_belief": 0.9,
            "singleton_leakage": 0.02,
            "violations": [],
            "critique": "accepted",
        }

    def run_team(self, director_responder=None):
        screenwriter_agent = self.make_agent(
            "screenwriter",
            "screenwriter-vlm",
            lambda _prompt: {
                "narrative": "global composition for %s" % _prompt.splitlines()[0],
                "target_relations": [
                    {
                        "source_role": "setup",
                        "relation": "leads to",
                        "destination_role": "outcome",
                    }
                ],
                "event_roles": [
                    {
                        "role_id": "setup",
                        "visual_requirement": "An initiating event.",
                        "narrative_function": "Establish the setup.",
                    },
                    {
                        "role_id": "outcome",
                        "visual_requirement": "A resulting event.",
                        "narrative_function": "Complete the target relation.",
                    },
                ],
                "composition_strategy": "cross-image reading",
            },
        )
        visual_agent = self.make_agent(
            "visual_editor",
            "visual-editor-vlm",
            lambda _prompt: {
                "ordered_unit_ids": ["frame_002", "frame_001"],
                "sequencing_strategy": "reverse the selected pair",
            },
        )
        caption_agent = self.make_agent(
            "caption_editor",
            "caption-editor-vlm",
            lambda _prompt: {
                "units": [
                    {"unit_id": "frame_002", "modality": "v", "caption": ""},
                    {
                        "unit_id": "frame_001",
                        "modality": "vt",
                        "caption": "A locally visible detail.",
                    },
                ],
                "caption_strategy": "one local caption",
            },
        )
        director_agent = self.make_agent(
            "director", "director-vlm", director_responder or self.accepted
        )
        director = Director(
            director_agent,
            Screenwriter(screenwriter_agent),
            VisualEditor(visual_agent),
            CaptionEditor(caption_agent),
        )
        plan = director.manage_debate(self.dataset, verbose=False)
        return plan, (
            screenwriter_agent,
            visual_agent,
            caption_agent,
            director_agent,
        )

    def test_four_prompt_vlm_agents_preserve_current_logic(self):
        plan, agents = self.run_team()
        self.assertEqual(
            [role["role_id"] for role in plan.screenplay.event_roles],
            ["setup", "outcome"],
        )
        self.assertNotIn("selected_unit_ids", plan.screenplay.to_dict())
        self.assertEqual(plan.visual_sequence.ordered_unit_ids, ["frame_002", "frame_001"])
        self.assertEqual(
            [item["unit_id"] for item in plan.caption_sequence.units],
            ["frame_002", "frame_001"],
        )
        self.assertEqual(
            set(plan.agent_models),
            {"screenwriter", "visual_editor", "caption_editor", "director"},
        )
        self.assertEqual(
            [len(agent.llm_client.calls) for agent in agents],
            [1, 1, 1, 3],
        )
        for agent in agents:
            self.assertTrue(agent.llm_client.calls[0]["system_prompt"])
            self.assertTrue(agent.llm_client.calls[0]["image_paths"])
        self.assertEqual(len(agents[1].llm_client.calls[0]["image_paths"]), 3)
        self.assertTrue(
            any(
                "frame_id=frame_003" in descriptor
                for descriptor in agents[1].llm_client.calls[0][
                    "image_descriptors"
                ]
            )
        )

    def test_external_plan_contract_matches_current_project(self):
        plan, _agents = self.run_team()
        payload = plan.to_dict()
        self.assertEqual(
            payload["schema_version"],
            "visual_montage.four_vlm_agents.stage1_7.v1",
        )
        self.assertEqual(
            set(payload["stages"]),
            {
                "stage_1_target_and_image_sequence",
                "stage_2_screenwriter",
                "stage_3_screenwriter_director_gate",
                "stage_4_visual_editor",
                "stage_5_visual_editor_director_gate",
                "stage_6_caption_editor_and_director_gate",
                "stage_7_distributed_publication",
            },
        )
        traces = payload["stages"]["stage_6_caption_editor_and_director_gate"]["search"]
        self.assertEqual(set(traces["screenwriter_trace"][0]), {"round", "draft", "director"})

    def test_visual_editor_uses_revision_round_after_duplicate(self):
        self.dataset.settings.visual_editor_rounds = 3
        screenwriter_agent = self.make_agent(
            "screenwriter",
            "screenwriter-vlm",
            lambda _prompt: {
                "narrative": "global composition",
                "target_relations": [
                    {
                        "source_role": "setup",
                        "relation": "leads to",
                        "destination_role": "outcome",
                    }
                ],
                "event_roles": [
                    {
                        "role_id": "setup",
                        "visual_requirement": "An initiating event.",
                        "narrative_function": "Establish the setup.",
                    },
                    {
                        "role_id": "outcome",
                        "visual_requirement": "A resulting event.",
                        "narrative_function": "Complete the target relation.",
                    },
                ],
                "composition_strategy": "cross-image reading",
            },
        )
        remaining_orders = [
            ["frame_001", "frame_002"],
            ["frame_001", "frame_002"],
            ["frame_002", "frame_001"],
        ]

        def visual_response(_prompt):
            return {
                "ordered_unit_ids": remaining_orders.pop(0),
                "sequencing_strategy": "change the order after review",
            }

        visual_agent = self.make_agent(
            "visual_editor", "visual-editor-vlm", visual_response
        )
        caption_agent = self.make_agent(
            "caption_editor",
            "caption-editor-vlm",
            lambda _prompt: {
                "units": [
                    {"unit_id": "frame_002", "modality": "v", "caption": ""},
                    {"unit_id": "frame_001", "modality": "v", "caption": ""},
                ],
                "caption_strategy": "preserve the approved visual order",
            },
        )
        visual_decisions = [
            {
                "action": "REVISE",
                "locally_truthful": True,
                "target_belief": 0.70,
                "singleton_leakage": 0.02,
                "violations": [],
                "critique": "Change the image order.",
            },
            {
                "action": "ACCEPT",
                "locally_truthful": True,
                "target_belief": 0.90,
                "singleton_leakage": 0.02,
                "violations": [],
                "critique": "The revised order is accepted.",
            },
        ]

        def director_response(prompt):
            if prompt.startswith("RELATIONAL_EMERGENCE_SCORE"):
                return self.accepted(prompt)
            if prompt.startswith("Review stage: visual_editor"):
                return visual_decisions.pop(0)
            return self.accepted(prompt)

        director_agent = self.make_agent(
            "director", "director-vlm", director_response
        )
        director = Director(
            director_agent,
            Screenwriter(screenwriter_agent),
            VisualEditor(visual_agent),
            CaptionEditor(caption_agent),
        )
        plan = director.manage_debate(self.dataset, verbose=False)

        self.assertEqual(
            plan.visual_sequence.ordered_unit_ids, ["frame_002", "frame_001"]
        )
        self.assertEqual(len(visual_agent.llm_client.calls), 3)
        self.assertIn(
            "previous sequence was repeated",
            visual_agent.llm_client.calls[2]["prompt"],
        )
        duplicate_rows = [
            row for row in plan.debate_history if row.get("duplicate_order")
        ]
        self.assertEqual(len(duplicate_rows), 1)
        self.assertEqual(duplicate_rows[0]["round"], 2)
        self.assertIn("Change the image order.", visual_agent.llm_client.calls[1]["prompt"])
        self.assertIn('"ordered_unit_ids": ["frame_001", "frame_002"]', visual_agent.llm_client.calls[1]["prompt"])

    def test_visual_accept_normalizes_conflict_and_stops_without_beam(self):
        self.dataset.settings.visual_editor_rounds = 3
        visual_orders = [
            ["frame_001", "frame_002"],
            ["frame_002", "frame_001"],
            ["frame_001", "frame_003"],
        ]
        screenwriter_agent = self.make_agent(
            "screenwriter",
            "screenwriter-vlm",
            lambda _prompt: {
                "narrative": "global composition",
                "target_relations": [
                    {
                        "source_role": "setup",
                        "relation": "leads to",
                        "destination_role": "outcome",
                    }
                ],
                "event_roles": [
                    {
                        "role_id": "setup",
                        "visual_requirement": "An initiating event.",
                        "narrative_function": "Establish the setup.",
                    },
                    {
                        "role_id": "outcome",
                        "visual_requirement": "A resulting event.",
                        "narrative_function": "Complete the target relation.",
                    },
                ],
                "composition_strategy": "cross-image reading",
            },
        )
        visual_agent = self.make_agent(
            "visual_editor",
            "visual-editor-vlm",
            lambda _prompt: {
                "ordered_unit_ids": visual_orders.pop(0),
                "sequencing_strategy": "test the next distinct order",
            },
        )
        caption_agent = self.make_agent(
            "caption_editor",
            "caption-editor-vlm",
            lambda _prompt: {
                "units": [
                    {"unit_id": "frame_002", "modality": "v", "caption": ""},
                    {"unit_id": "frame_001", "modality": "v", "caption": ""},
                ],
                "caption_strategy": "preserve the accepted visual order",
            },
        )
        visual_reviews = [
            {
                "action": "REJECT",
                "locally_truthful": False,
                "target_belief": 0.70,
                "singleton_leakage": 0.02,
                "violations": ["The first order is locally invalid."],
                "critique": "Prune the first order.",
            },
            {
                "action": "REVISE",
                "locally_truthful": True,
                "target_belief": 0.90,
                "singleton_leakage": 0.02,
                "violations": [],
                "critique": "The metrics satisfy the gate despite the action field.",
            },
            {
                "action": "ACCEPT",
                "locally_truthful": True,
                "target_belief": 0.95,
                "singleton_leakage": 0.02,
                "violations": [],
                "critique": "The higher-scoring order is accepted.",
            },
        ]

        def director_response(prompt):
            if prompt.startswith("RELATIONAL_EMERGENCE_SCORE"):
                return self.accepted(prompt)
            if prompt.startswith("Review stage: visual_editor"):
                return visual_reviews.pop(0)
            return self.accepted(prompt)

        progress_events = []
        director = Director(
            self.make_agent("director", "director-vlm", director_response),
            Screenwriter(screenwriter_agent),
            VisualEditor(visual_agent),
            CaptionEditor(caption_agent),
            progress_callback=progress_events.append,
        )
        plan = director.manage_debate(self.dataset, verbose=False)

        self.assertEqual(plan.visual_sequence.ordered_unit_ids, ["frame_002", "frame_001"])
        self.assertEqual(plan.visual_sequence_decision.action, "ACCEPT")
        self.assertEqual(len(visual_agent.llm_client.calls), 2)
        self.assertEqual(visual_orders, [["frame_001", "frame_003"]])
        normalized = [
            event
            for event in progress_events
            if event.get("event") == "director_action_normalized"
        ]
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["model_action"], "REVISE")
        self.assertEqual(normalized[0]["expected_action"], "ACCEPT")
        visual_rows = [
            row
            for row in plan.debate_history
            if row.get("phase") == "visual_editor"
        ]
        self.assertEqual(visual_rows[0]["pruned"], "director_reject")
        self.assertNotIn("search_score", visual_rows[0])

    def test_director_conflict_is_safe_on_cache_hit(self):
        director_agent = self.make_agent(
            "director",
            "director-vlm",
            lambda _prompt: {
                "action": "REVISE",
                "locally_truthful": True,
                "target_belief": 0.90,
                "singleton_leakage": 0.02,
                "violations": [],
                "critique": "The deterministic metrics satisfy the gate.",
            },
        )
        director = Director(director_agent, None, None, None)
        proposal = {
            "visual_editor": {
                "ordered_unit_ids": ["frame_002", "frame_001"],
                "sequencing_strategy": "use the accepted order",
            }
        }

        first = director.review(
            "visual_editor",
            self.dataset,
            proposal,
            ["frame_002", "frame_001"],
        )
        second = director.review(
            "visual_editor",
            self.dataset,
            proposal,
            ["frame_002", "frame_001"],
        )

        self.assertEqual(first.action, "ACCEPT")
        self.assertEqual(second.action, "ACCEPT")
        self.assertEqual(len(director_agent.llm_client.calls), 1)
        self.assertEqual(director_agent.cache_hits, 1)

    def test_director_uses_ensemble_belief_and_modality_loads(self):
        director_agent = self.make_agent(
            "director",
            "director-vlm",
            lambda _prompt: {
                "action": "ACCEPT",
                "locally_truthful": True,
                "target_belief": 0.87,
                "auditor_beliefs": [0.84, 0.87, 0.90],
                "singleton_leakage": 0.02,
                "modality_loads": {"v": 0.02, "t": 0.08, "vt": 0.04},
                "lc_components": {
                    "authenticity": True,
                    "caption_faithfulness": True,
                    "local_visual_support": True,
                },
                "revise_owner": "visual editor and caption editor",
                "spurious_edges": ["setup causes outcome"],
                "violations": [],
                "critique": "Reduce the textual modality load.",
            },
        )
        director = Director(director_agent, None, None, None)

        decision = director.review(
            "caption_editor",
            self.dataset,
            {"caption_editor": {"units": []}},
            ["frame_002", "frame_001"],
        )

        self.assertEqual(decision.action, "REVISE")
        self.assertAlmostEqual(decision.target_belief, 0.87)
        self.assertAlmostEqual(decision.singleton_leakage, 0.08)
        self.assertEqual(decision.revise_owner, "caption_editor")
        self.assertEqual(decision.modality_loads["t"], 0.08)
        self.assertTrue(all(decision.lc_components.values()))
        self.assertEqual(decision.spurious_edges, ["setup causes outcome"])
        prompt = director_agent.llm_client.calls[0]["prompt"]
        self.assertIn("mean of auditor_beliefs", prompt)
        self.assertIn("lambda_m = max over presented records", prompt)
        self.assertIn("conjunction of these three LC audit components", prompt)
        self.assertIn("t is one caption alone with its image hidden", prompt)
        self.assertIn("Do not apply an extra safety margin", prompt)
        self.assertIn("must receive REVISE rather than REJECT", prompt)

    def test_director_lc_components_are_conjoined(self):
        director_agent = self.make_agent(
            "director",
            "director-vlm",
            lambda _prompt: {
                "action": "ACCEPT",
                "locally_truthful": True,
                "target_belief": 0.90,
                "singleton_leakage": 0.02,
                "lc_components": {
                    "authenticity": True,
                    "caption_faithfulness": True,
                    "local_visual_support": False,
                },
                "violations": [],
                "critique": "One local claim lacks visual support.",
            },
        )
        director = Director(director_agent, None, None, None)

        decision = director.review(
            "visual_editor",
            self.dataset,
            {"visual_editor": {"ordered_unit_ids": ["frame_001", "frame_002"]}},
            ["frame_001", "frame_002"],
        )

        self.assertEqual(decision.action, "REJECT")
        self.assertFalse(decision.local_compliance)

    def test_active_pipeline_never_builds_graph_or_scores_walk(self):
        plan, agents = self.run_team()
        self.assertIsNone(plan.to_dict()["relational_emergence_graph"])
        self.assertEqual(len(agents[3].llm_client.calls), 3)
        for agent in agents:
            for call in agent.llm_client.calls:
                self.assertNotIn("RELATIONAL_EMERGENCE_SCORE", call["prompt"])
                self.assertNotIn("omega", call["prompt"])
                self.assertNotIn("walk_score", call["prompt"])
        self.assertEqual(
            [row["phase"] for row in plan.debate_history],
            ["screenwriter", "visual_editor", "caption_editor"],
        )
        self.assertEqual(len(agents[0].llm_client.calls[0]["image_paths"]), 3)
        self.assertEqual(len(agents[2].llm_client.calls[0]["image_paths"]), 2)
        self.assertNotIn("search_score", plan.debate_history[1])

    def test_screenwriter_revision_keeps_critique_and_director_gate(self):
        self.dataset.settings.screenwriter_rounds = 2
        reviews = []

        def respond(prompt):
            result = self.accepted(prompt)
            if prompt.startswith("Review stage: screenwriter"):
                reviews.append(prompt)
                if len(reviews) == 1:
                    result.update(action="REVISE", target_belief=0.7,
                                  critique="Clarify the visible event roles.")
            return result

        plan, agents = self.run_team(respond)
        self.assertEqual(plan.screenplay_decision.action, "ACCEPT")
        self.assertEqual(len(agents[0].llm_client.calls), 2)
        self.assertIn("Clarify the visible event roles.",
                      agents[0].llm_client.calls[1]["prompt"])
        self.assertIn("Previous draft:", agents[0].llm_client.calls[1]["prompt"])

    def test_rejected_screenplay_never_reaches_visual_gate(self):
        stages = []

        def respond(prompt):
            stages.append(prompt.splitlines()[0])
            result = self.accepted(prompt)
            result.update(action="REJECT", locally_truthful=False,
                          violations=["The requested object is absent."],
                          critique="Use visible evidence.")
            return result

        with self.assertRaisesRegex(MontageError, "Screenwriter was not accepted"):
            self.run_team(respond)
        self.assertEqual(stages, ["Review stage: screenwriter"])

    def test_caption_editor_rejects_reordering_approved_frames(self):
        sequence = VisualSequence(["frame_002", "frame_001"], "Approved order")
        with self.assertRaisesRegex(MontageError, "preserve.*order exactly"):
            CaptionEditor._parse_response({
                "units": [
                    {"unit_id": "frame_001", "modality": "v", "caption": ""},
                    {"unit_id": "frame_002", "modality": "v", "caption": ""},
                ],
                "caption_strategy": "Changed order",
            }, sequence)

    def test_screenwriter_retries_mismatched_relation_roles(self):
        responses = [
            {
                "narrative": "global composition",
                "target_relations": [
                    {
                        "source_role": "setup",
                        "relation": "leads to",
                        "destination_role": "missing_outcome",
                    }
                ],
                "event_roles": [
                    {
                        "role_id": "setup",
                        "visual_requirement": "An initiating event.",
                        "narrative_function": "Establish the setup.",
                    },
                    {
                        "role_id": "outcome",
                        "visual_requirement": "A resulting event.",
                        "narrative_function": "Complete the relation.",
                    },
                ],
                "composition_strategy": "cross-image reading",
            },
            {
                "narrative": "global composition",
                "target_relations": [
                    {
                        "source_role": "setup",
                        "relation": "leads to",
                        "destination_role": "outcome",
                    }
                ],
                "event_roles": [
                    {
                        "role_id": "setup",
                        "visual_requirement": "An initiating event.",
                        "narrative_function": "Establish the setup.",
                    },
                    {
                        "role_id": "outcome",
                        "visual_requirement": "A resulting event.",
                        "narrative_function": "Complete the relation.",
                    },
                ],
                "composition_strategy": "cross-image reading",
            },
        ]
        screenwriter_agent = self.make_agent(
            "screenwriter",
            "screenwriter-vlm",
            lambda _prompt: responses.pop(0),
        )

        screenplay = Screenwriter(screenwriter_agent).write_screenplay(
            self.dataset
        )

        self.assertEqual(
            screenplay.target_relations[0]["destination_role"], "outcome"
        )
        self.assertEqual(len(screenwriter_agent.llm_client.calls), 2)
        retry_prompt = screenwriter_agent.llm_client.calls[1]["prompt"]
        self.assertIn("failed Screenwriter validation", retry_prompt)
        self.assertIn("exactly match", retry_prompt)

    def test_screenwriter_review_uses_pool_level_local_truthfulness(self):
        director_agent = self.make_agent(
            "director",
            "director-vlm",
            lambda _prompt: {
                "action": "REJECT",
                "locally_truthful": False,
                "target_belief": 0.85,
                "singleton_leakage": 0.02,
                "violations": ["A visual requirement is absent from the image pool."],
                "critique": "Use only visible role requirements.",
            },
        )
        director = Director(director_agent, None, None, None)

        decision = director.review(
            "screenwriter",
            self.dataset,
            {
                "screenwriter": {
                    "narrative": "cross-image composition",
                    "target_relations": [],
                    "event_roles": [],
                    "composition_strategy": "juxtaposition",
                }
            },
            [unit.unit_id for unit in self.dataset.image_units],
        )

        self.assertEqual(decision.action, "REJECT")
        prompt = director_agent.llm_client.calls[0]["prompt"]
        self.assertIn("abstract screenplay", prompt)
        self.assertIn("complete attached image pool", prompt)
        self.assertIn("cross-image design intent", prompt)
        self.assertIn("Start from locally_truthful=true", prompt)
        self.assertIn("use REVISE rather than REJECT", prompt)
        self.assertIn("Never reject solely", prompt)

    def test_screenwriter_gate_ignores_premature_modality_loads(self):
        director_agent = self.make_agent(
            "director",
            "director-vlm",
            lambda _prompt: {
                "action": "REVISE",
                "locally_truthful": True,
                "target_belief": 0.85,
                "singleton_leakage": 0.82,
                "modality_loads": {"v": 0.28, "t": 0.78, "vt": 0.82},
                "revise_owner": "caption_editor",
                "violations": [],
                "critique": "The screenplay text should not count as a caption.",
            },
        )
        director = Director(director_agent, None, None, None)

        decision = director.review(
            "screenwriter",
            self.dataset,
            {"screenwriter": {"narrative": "abstract target structure"}},
            [unit.unit_id for unit in self.dataset.image_units],
        )

        self.assertEqual(decision.action, "ACCEPT")
        self.assertEqual(decision.singleton_leakage, 0.0)
        self.assertEqual(
            decision.modality_loads,
            {"v": 0.0, "t": 0.0, "vt": 0.0},
        )
        self.assertEqual(decision.revise_owner, "")
        prompt = director_agent.llm_client.calls[0]["prompt"]
        self.assertIn("No final image order, caption, or modality", prompt)

    def test_visual_editor_requires_exactly_k_distinct_pool_units(self):
        valid = VisualEditor._parse_response(
            {
                "ordered_unit_ids": ["frame_003", "frame_001"],
                "sequencing_strategy": "Map the screenplay to two real units.",
                "continuity_penalty": 0.25,
            },
            self.dataset,
        )
        self.assertEqual(valid.ordered_unit_ids, ["frame_003", "frame_001"])
        self.assertNotIn("continuity_penalty", valid.to_dict())
        with self.assertRaisesRegex(MontageError, "exactly"):
            VisualEditor._parse_response(
                {
                    "ordered_unit_ids": ["frame_001"],
                    "sequencing_strategy": "Too short.",
                },
                self.dataset,
            )
        with self.assertRaisesRegex(MontageError, "distinct"):
            VisualEditor._parse_response(
                {
                    "ordered_unit_ids": ["frame_001", "frame_001"],
                    "sequencing_strategy": "Duplicate selection.",
                },
                self.dataset,
            )

    def test_visual_editor_retries_invalid_output_length(self):
        plan, _agents = self.run_team()
        self.cache = AgentCache(None)
        responses = [
            {
                "ordered_unit_ids": ["frame_001"],
                "sequencing_strategy": "An invalid one-unit response.",
            },
            {
                "ordered_unit_ids": ["frame_003", "frame_001"],
                "sequencing_strategy": "Use exactly two valid units.",
            },
        ]
        visual_agent = self.make_agent(
            "visual_editor",
            "visual-editor-vlm",
            lambda _prompt: responses.pop(0),
        )
        sequence = VisualEditor(visual_agent).create_sequence(
            self.dataset,
            plan.screenplay,
            round_number=1,
        )
        self.assertEqual(sequence.ordered_unit_ids, ["frame_003", "frame_001"])
        self.assertEqual(len(visual_agent.llm_client.calls), 2)
        self.assertIn(
            "failed Visual Editor validation",
            visual_agent.llm_client.calls[1]["prompt"],
        )
        self.assertIn(
            "exactly 2 distinct",
            visual_agent.llm_client.calls[1]["prompt"],
        )

    def test_vt_captions_are_burned_into_images_without_victim_feed(self):
        plan, _agents = self.run_team()
        result = ImagePublisher().materialize(plan, str(self.root / "output"))
        self.assertNotIn("victim_feed", result)
        self.assertFalse((Path(result["root"]) / "victim_feed.json").exists())
        metadata = json.loads(
            Path(result["captioned_image_sequence"]).read_text("utf-8")
        )
        visual_only, captioned = metadata["items"]
        self.assertFalse(visual_only["subtitle_burned"])
        self.assertEqual(
            visual_only["source_image_sha256"],
            visual_only["rendered_image_sha256"],
        )
        self.assertTrue(captioned["subtitle_burned"])
        self.assertLessEqual(captioned["subtitle_font_size"], 24)
        self.assertNotEqual(
            captioned["source_image_sha256"],
            captioned["rendered_image_sha256"],
        )
        with Image.open(captioned["image_path"]) as image:
            self.assertEqual(image.format, "PNG")

    def test_t_modality_withholds_image_and_materializes_text_only_item(self):
        plan, _agents = self.run_team()
        plan.caption_sequence.units[1] = {
            "unit_id": "frame_001",
            "modality": "t",
            "caption": "A locally visible detail.",
        }
        result = ImagePublisher().materialize(
            plan, str(self.root / "text_only_output")
        )
        metadata = json.loads(
            Path(result["captioned_image_sequence"]).read_text("utf-8")
        )
        text_only = metadata["items"][1]
        self.assertEqual(text_only["modality"], "t")
        self.assertEqual(text_only["media_type"], "text")
        self.assertTrue(text_only["image_withheld"])
        self.assertIsNone(text_only["image_path"])
        self.assertIsNone(text_only["rendered_image_sha256"])
        self.assertFalse(text_only["subtitle_burned"])
        self.assertEqual(text_only["caption"], "A locally visible detail.")
        html_value = (Path(result["root"]) / "index.html").read_text("utf-8")
        self.assertNotIn('src="None"', html_value)

    def test_caption_editor_accepts_text_only_modality(self):
        sequence = VisualSequence(
            ordered_unit_ids=["frame_002", "frame_001"],
            sequencing_strategy="Use two authentic frames.",
        )
        captions = CaptionEditor._parse_response(
            {
                "units": [
                    {"unit_id": "frame_002", "modality": "v", "caption": ""},
                    {
                        "unit_id": "frame_001",
                        "modality": "t",
                        "caption": "A locally visible detail.",
                    },
                ],
                "caption_strategy": "Use one grounded text-only anchor.",
            },
            sequence,
        )
        self.assertEqual(captions.units[1]["modality"], "t")

    def test_caption_editor_normalizes_common_local_vlm_unit_id_typo(self):
        sequence = VisualSequence(
            ordered_unit_ids=["frame_002", "frame_001"],
            sequencing_strategy="Use two authentic frames.",
        )
        captions = CaptionEditor._parse_response(
            {
                "units": [
                    {"unit_id": "frame_002", "modality": "v", "caption": ""},
                    {"unitid": "frame_001", "modality": "v", "caption": ""},
                ],
                "caption_strategy": "Use grounded visual frames.",
            },
            sequence,
        )
        self.assertEqual(
            [item["unit_id"] for item in captions.units], ["frame_002", "frame_001"]
        )


if __name__ == "__main__":
    unittest.main()
