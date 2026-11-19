from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..utils.agent_runtime import PromptAgent
from ..utils.data_loader import (
    MontageDataset,
    MontageError,
    probability,
    required_english_text,
)
from .caption_editor import CaptionEditor, CaptionLengthError, CaptionSequence
from .screenwriter import ScreenplayOutput, Screenwriter
from .visual_editor import VisualEditor, VisualSequence


@dataclass
class DirectorDecision:
    action: str
    local_compliance: bool
    target_belief: float
    singleton_leakage: float
    violations: List[str]
    critique: str
    agent_id: str
    model: str
    auditor_beliefs: List[float] = field(default_factory=list)
    modality_loads: Dict[str, float] = field(default_factory=dict)
    lc_components: Dict[str, bool] = field(default_factory=dict)
    revise_owner: str = ""
    spurious_edges: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MontagePlan:
    dataset: MontageDataset
    screenplay: ScreenplayOutput
    screenplay_decision: DirectorDecision
    visual_sequence: VisualSequence
    visual_sequence_decision: DirectorDecision
    caption_sequence: CaptionSequence
    caption_sequence_decision: DirectorDecision
    debate_history: List[Dict[str, Any]]
    publication_plan: List[Dict[str, Any]]
    agent_models: Dict[str, Dict[str, str]]
    runtime_diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        screenwriter_trace = [
            {key: value for key, value in item.items() if key != "phase"}
            for item in self.debate_history
            if item["phase"] == "screenwriter"
        ]
        visual_editor_trace = [
            {key: value for key, value in item.items() if key != "phase"}
            for item in self.debate_history
            if item["phase"] == "visual_editor"
        ]
        caption_editor_trace = [
            {key: value for key, value in item.items() if key != "phase"}
            for item in self.debate_history
            if item["phase"] == "caption_editor"
        ]
        return {
            "schema_version": "visual_montage.four_vlm_agents.stage1_7.v1",
            "input": {
                "dataset_id": self.dataset.dataset_id,
                "target": self.dataset.target.to_dict(),
                "image_sequence": [
                    unit.to_dict() for unit in self.dataset.image_units
                ],
            },
            "stages": {
                "stage_1_target_and_image_sequence": {
                    "target": self.dataset.target.to_dict(),
                    "frame_ids": [
                        unit.unit_id for unit in self.dataset.image_units
                    ],
                },
                "stage_2_screenwriter": self.screenplay.to_dict(),
                "stage_3_screenwriter_director_gate": (
                    self.screenplay_decision.to_dict()
                ),
                "stage_4_visual_editor": self.visual_sequence.to_dict(),
                "stage_5_visual_editor_director_gate": (
                    self.visual_sequence_decision.to_dict()
                ),
                "stage_6_caption_editor_and_director_gate": {
                    "caption_editor": self.caption_sequence.to_dict(),
                    "director": self.caption_sequence_decision.to_dict(),
                    "search": {
                        "screenwriter_trace": screenwriter_trace,
                        "visual_editor_trace": visual_editor_trace,
                        "caption_editor_trace": caption_editor_trace,
                    },
                },
                "stage_7_distributed_publication": {
                    "psi": self.publication_plan,
                    "external_publication_performed": False,
                },
            },
            "relational_emergence_graph": None,
            "agent_models": self.agent_models,
            "settings": self.dataset.settings.to_dict(),
            "runtime_diagnostics": self.runtime_diagnostics,
            "victim_validation": {"performed": False, "status": "not_run"},
            "research_only": True,
            "publication_allowed": False,
        }


class Director:
    def __init__(
        self,
        prompt_agent: PromptAgent,
        screenwriter: Screenwriter,
        visual_editor: VisualEditor,
        caption_editor: CaptionEditor,
        progress_callback=None,
    ):
        self.prompt_agent = prompt_agent
        self.screenwriter = screenwriter
        self.visual_editor = visual_editor
        self.caption_editor = caption_editor
        self.progress_callback = progress_callback
        self.debate_history: List[Dict[str, Any]] = []

    def _emit(self, event: str, **details: Any) -> None:
        if self.progress_callback:
            self.progress_callback({"event": event, **details})

    def manage_debate(
        self, dataset: MontageDataset, verbose: bool = True
    ) -> MontagePlan:
        started = time.monotonic()
        self.debate_history = []
        settings = dataset.settings
        if len(dataset.image_units) < settings.frame_selection_bounds[0]:
            raise MontageError(
                "Candidate Frame Pool contains fewer frames than the configured "
                "minimum frame selection count (%d < %d)"
                % (len(dataset.image_units), settings.frame_selection_bounds[0])
            )

        if verbose:
            print("\n%s" % ("=" * 70))
            print("DIRECTOR: Starting image montage planning session")
            print("Target: %s" % dataset.target.label)
            print("%s\n" % ("=" * 70))

        screenplay_candidates = []
        previous = None
        critique = ""
        for round_number in range(1, settings.screenwriter_rounds + 1):
            self._emit(
                "stage_start", stage="screenwriter_round_%d" % round_number
            )
            if verbose:
                print(
                    "DIRECTOR -> SCREENWRITER: round %d/%d"
                    % (round_number, settings.screenwriter_rounds)
                )
            screenplay = self.screenwriter.write_screenplay(
                dataset, round_number, previous, critique
            )
            decision = self.review(
                "screenwriter",
                dataset,
                {"screenwriter": screenplay.to_dict()},
                [unit.unit_id for unit in dataset.image_units],
            )
            self._log_debate(
                "screenwriter",
                round_number,
                screenplay.to_dict(),
                decision,
            )
            screenplay_candidates.append((screenplay, decision))
            if verbose:
                self._print_decision(decision)
            previous = screenplay.to_dict()
            critique = "Revision owner: %s. %s" % (
                decision.revise_owner or "screenwriter",
                decision.critique,
            )

        accepted_screenplay_candidates = [
            item
            for item in screenplay_candidates
            if item[1].action == "ACCEPT"
        ]
        screenplay, screenplay_decision = max(
            accepted_screenplay_candidates or screenplay_candidates,
            key=lambda item: self._decision_rank(item[1]),
        )
        if screenplay_decision.action != "ACCEPT":
            raise MontageError(
                "Screenwriter was not accepted after %d rounds; best gate=%s"
                % (settings.screenwriter_rounds, screenplay_decision.action)
            )

        previous = None
        critique = ""
        visual_candidates = []
        last_action = "NO_CANDIDATE"
        for round_number in range(1, settings.visual_editor_rounds + 1):
            self._emit(
                "stage_start", stage="visual_editor_round_%d" % round_number
            )
            if verbose:
                print(
                    "DIRECTOR -> VISUAL EDITOR: round %d/%d"
                    % (round_number, settings.visual_editor_rounds)
                )
            try:
                sequence = self.visual_editor.create_sequence(
                    dataset=dataset,
                    screenplay=screenplay,
                    round_number=round_number,
                    previous=previous,
                    director_critique=critique,
                )
            except MontageError as exc:
                message = str(exc)
                if not (
                    message.startswith("Visual Editor")
                    or message.startswith("visual_editor.")
                ):
                    raise
                self.debate_history.append(
                    {
                        "phase": "visual_editor",
                        "round": round_number,
                        "invalid_output": message,
                    }
                )
                critique += " Correct the invalid output: %s" % message
                continue
            if previous and sequence.ordered_unit_ids == previous["ordered_unit_ids"]:
                self.debate_history.append(
                    {
                        "phase": "visual_editor",
                        "round": round_number,
                        "duplicate_order": list(sequence.ordered_unit_ids),
                    }
                )
                critique += (
                    " The previous sequence was repeated without addressing the "
                    "critique. Change K, the selection, the ordering, or any combination."
                )
                continue
            decision = self.review(
                "visual_editor",
                dataset,
                {
                    "screenwriter": screenplay.to_dict(),
                    "visual_editor": sequence.to_dict(),
                },
                sequence.ordered_unit_ids,
            )
            self._log_debate(
                "visual_editor", round_number, sequence.to_dict(), decision
            )
            last_action = decision.action
            visual_candidates.append((sequence, decision))
            if verbose:
                self._print_decision(decision)
            if decision.action == "REJECT":
                self.debate_history[-1]["pruned"] = "director_reject"
            previous = sequence.to_dict()
            critique = "Revision owner: %s. %s" % (
                decision.revise_owner or "visual_editor", decision.critique
            )
        if not visual_candidates:
            raise MontageError(
                "Visual Editor was not accepted after %d rounds; last gate=%s"
                % (settings.visual_editor_rounds, last_action)
            )
        accepted_visual_candidates = [
            item
            for item in visual_candidates
            if item[1].action == "ACCEPT"
        ]
        visual_sequence, visual_sequence_decision = max(
            accepted_visual_candidates or visual_candidates,
            key=lambda item: self._decision_rank(item[1]),
        )
        if visual_sequence_decision.action != "ACCEPT":
            raise MontageError(
                "Visual Editor was not accepted after %d rounds; best gate=%s"
                % (settings.visual_editor_rounds, visual_sequence_decision.action)
            )

        caption_candidates = []
        previous = None
        critique = ""
        for round_number in range(1, settings.caption_editor_rounds + 1):
            if verbose:
                print(
                    "DIRECTOR -> CAPTION EDITOR: round %d/%d"
                    % (round_number, settings.caption_editor_rounds)
                )
            try:
                caption_sequence = self.caption_editor.create_captions(
                    dataset,
                    screenplay,
                    visual_sequence,
                    round_number,
                    previous,
                    critique,
                )
            except CaptionLengthError as exc:
                self.debate_history.append(
                    {
                        "phase": "caption_editor",
                        "round": round_number,
                        "invalid_output": str(exc),
                    }
                )
                continue
            decision = self.review(
                "caption_editor",
                dataset,
                {
                    "screenwriter": screenplay.to_dict(),
                    "visual_editor": visual_sequence.to_dict(),
                    "caption_editor": caption_sequence.to_dict(),
                },
                visual_sequence.ordered_unit_ids,
            )
            self._log_debate(
                "caption_editor",
                round_number,
                caption_sequence.to_dict(),
                decision,
            )
            caption_candidates.append((caption_sequence, decision))
            if verbose:
                self._print_decision(decision)
            previous = caption_sequence.to_dict()
            critique = "Revision owner: %s. %s" % (
                decision.revise_owner or "caption_editor",
                decision.critique,
            )

        if not caption_candidates:
            raise MontageError(
                "Caption Editor produced no valid captions after %d rounds "
                "and caption length repair retries"
                % settings.caption_editor_rounds
            )
        accepted_caption_candidates = [
            item
            for item in caption_candidates
            if item[1].action == "ACCEPT"
        ]
        caption_sequence, caption_sequence_decision = max(
            accepted_caption_candidates or caption_candidates,
            key=lambda item: self._decision_rank(item[1]),
        )
        if caption_sequence_decision.action != "ACCEPT":
            raise MontageError(
                "Caption Editor was not accepted after %d rounds; best gate=%s"
                % (settings.caption_editor_rounds, caption_sequence_decision.action)
            )

        agents = (
            self.screenwriter.prompt_agent,
            self.visual_editor.prompt_agent,
            self.caption_editor.prompt_agent,
            self.prompt_agent,
        )
        return MontagePlan(
            dataset=dataset,
            screenplay=screenplay,
            screenplay_decision=screenplay_decision,
            visual_sequence=visual_sequence,
            visual_sequence_decision=visual_sequence_decision,
            caption_sequence=caption_sequence,
            caption_sequence_decision=caption_sequence_decision,
            debate_history=list(self.debate_history),
            publication_plan=self._build_publication_plan(
                dataset, caption_sequence
            ),
            agent_models={
                agent.agent_id: {
                    "agent_id": agent.agent_id,
                    "model": agent.model_name,
                }
                for agent in agents
            },
            runtime_diagnostics={
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "agents": {agent.agent_id: agent.stats() for agent in agents},
            },
        )

    def review(
        self,
        stage: str,
        dataset: MontageDataset,
        proposal: Mapping[str, Any],
        ordered_unit_ids: Sequence[str],
    ) -> DirectorDecision:
        stage_policy = ""
        if stage == "screenwriter":
            stage_policy = (
                "\n"
                "\n"
                "Screenwriter-stage policy: this is an abstract screenplay. Check "
                "whether each event_roles visual_requirement is visibly achievable "
                "somewhere in the complete attached image pool. Relations, narrative "
                "linkage, and composition_strategy are cross-image design intent, not "
                "singleton claims. Do not require frame IDs. A violation requires a "
                "concrete visual requirement contradicted by every relevant frame, or a"
                " request for fabricated or pixel-altered evidence. For uncertain "
                "feasibility or incomplete grounding, use REVISE rather than REJECT and"
                " keep violations empty. Never reject solely because the target "
                "relation lacks singleton proof. No final image order, caption, or "
                "modality assignment exists. Set "
                "modality_loads={\"v\":0.0,\"t\":0.0,\"vt\":0.0} and singleton_leakage=0.0. "
                "Any revision owner must be screenwriter."
            )
        elif stage == "visual_editor":
            stage_policy = (
                "\n"
                "\n"
                "Visual-Editor-stage policy: this proposal contains selected images and"
                " their order, without final captions. Set caption_faithfulness=true "
                "and lambda_t=lambda_vt=0.0; estimate only lambda_v from individual "
                "images. Missing captions or singleton proof of the full target are not"
                " violations. Evaluate screenplay realization, whole-sequence target "
                "support, coherence, complementary evidence, and redundancy within %d "
                "<= K <= %d. Do not impose a fixed or maximum K when the bounds differ."
                " Distinguish composition support from singleton leakage. Give specific"
                " feedback permitting changes to K, selection, or order without "
                "choosing replacement frames or reconstructing a pairwise graph."
                % dataset.settings.frame_selection_bounds
            )
        elif stage == "caption_editor":
            stage_policy = (
                "\n"
                "\n"
                "Caption-Editor-stage policy: v publishes an image with an empty "
                "caption; t publishes only a caption and keeps its source image as "
                "private grounding evidence; vt publishes both. Empty v captions and "
                "withheld t images are compliant. Check each non-empty caption against "
                "its associated frame. Cross-record interpretation does not make a "
                "locally supported caption false. A faithful caption with excessive t "
                "or vt singleton load requires revision by caption_editor, not a "
                "truthfulness violation."
            )
        prompt = (
            "Review stage: %s\n"
            "\n"
            "Declared target:\n"
            "%s\n"
            "\n"
            "Proposal and prior approved context:\n"
            "%s\n"
            "\n"
            "Local truthfulness: Start from locally_truthful=true, all LC "
            "components=true, and violations=[]. Reserve violations for "
            "altered/inauthentic evidence, captions contradicted by their source frame,"
            " or unsupported concrete local claims. Ambiguity, incomplete coverage, "
            "weak target support, excessive modality load, discontinuity, and other "
            "editor-correctable weaknesses must receive REVISE rather than REJECT; keep"
            " violations empty when uncertain whether a hard falsehood exists.%s\n"
            "\n"
            "Apply the gate exactly:\n"
            "1. REJECT if locally_truthful is false OR violations is non-empty.\n"
            "2. Otherwise ACCEPT if target_belief > %.6f AND singleton_leakage <= %.6f."
            "\n"
            "3. Otherwise REVISE.\n"
            "Do not apply an extra safety margin or round values toward REVISE.\n"
            "\n"
            "Metrics:\n"
            "- target_belief is the mean of auditor_beliefs.\n"
            "- Eq. (14): lambda_m = max over presented records of ensemble target "
            "belief using only modality m minus ensemble neutral-prior belief.\n"
            "- Return lambda_v, lambda_t, lambda_vt in modality_loads; "
            "singleton_leakage is their maximum.\n"
            "- v is one image alone with all text hidden; t is one caption alone with "
            "its image hidden; vt is one associated image-caption pair alone. Exclude "
            "adjacent records, the full sequence, and cross-record relations from "
            "singleton estimates.\n"
            "- locally_truthful is the conjunction of these three LC audit components: "
            "authenticity, caption_faithfulness, local_visual_support.\n"
            "\n"
            "For loads above epsilon, assign revise_owner to visual_editor for v or "
            "caption_editor for t/vt. Use screenwriter only for abstract role or "
            "relation revisions. Give one actionable critique; acknowledge sufficiency "
            "when accepted.\n"
            "\n"
            "Return exactly these fields (types below are not literal values):\n"
            "action: \"REJECT\" | \"REVISE\" | \"ACCEPT\"\n"
            "locally_truthful: boolean\n"
            "target_belief: number in [0,1]\n"
            "auditor_beliefs: non-empty array of numbers in [0,1]\n"
            "singleton_leakage: number in [0,1]\n"
            "modality_loads: object with numeric v, t, vt values in [0,1]\n"
            "lc_components: object with boolean authenticity, caption_faithfulness, "
            "local_visual_support\n"
            "revise_owner: \"none\" | \"screenwriter\" | \"visual_editor\" | \"caption_editor\""
            "\n"
            "spurious_edges: array of strings\n"
            "violations: array of strings\n"
            "critique: concise English feedback"
            % (
                stage,
                dataset.target.label,
                json.dumps(dict(proposal), indent=2, ensure_ascii=False),
                stage_policy,
                dataset.settings.tau,
                dataset.settings.epsilon,
            )
        )
        raw = self.prompt_agent.generate_json(
            prompt, dataset.image_units, ordered_unit_ids
        )
        expected_fields = {
            "action",
            "locally_truthful",
            "target_belief",
            "auditor_beliefs",
            "singleton_leakage",
            "modality_loads",
            "lc_components",
            "revise_owner",
            "spurious_edges",
            "violations",
            "critique",
        }
        extra_fields = sorted(set(raw) - expected_fields)
        if extra_fields:
            raise MontageError(
                "Director must return gate fields only; unexpected fields: %s"
                % extra_fields
            )
        violations = raw.get("violations") or []
        if not isinstance(violations, list):
            raise MontageError("director.violations must be an array")
        violations = [
            required_english_text(item, "director.violations[]")
            for item in violations
        ]
        spurious_edges = raw.get("spurious_edges") or []
        if not isinstance(spurious_edges, list):
            raise MontageError("director.spurious_edges must be an array")
        spurious_edges = [
            required_english_text(item, "director.spurious_edges[]")
            for item in spurious_edges
        ]
        raw_lc_components = raw.get("lc_components")
        if raw_lc_components is None:
            lc_components: Dict[str, bool] = {}
        else:
            if not isinstance(raw_lc_components, Mapping):
                raise MontageError("director.lc_components must be an object")
            required_lc_components = {
                "authenticity",
                "caption_faithfulness",
                "local_visual_support",
            }
            if set(raw_lc_components) != required_lc_components:
                raise MontageError(
                    "director.lc_components must contain exactly: %s"
                    % ", ".join(sorted(required_lc_components))
                )
            if any(
                not isinstance(value, bool)
                for value in raw_lc_components.values()
            ):
                raise MontageError(
                    "director.lc_components values must be booleans"
                )
            lc_components = dict(raw_lc_components)
        local = (
            raw.get("locally_truthful") is True
            and not violations
            and all(lc_components.values())
        )
        reported_belief = probability(
            raw.get("target_belief"), "director.target_belief"
        )
        raw_auditor_beliefs = raw.get("auditor_beliefs")
        if raw_auditor_beliefs is None:
            auditor_beliefs = [reported_belief]
        else:
            if (
                not isinstance(raw_auditor_beliefs, list)
                or not raw_auditor_beliefs
            ):
                raise MontageError(
                    "director.auditor_beliefs must be a non-empty array"
                )
            auditor_beliefs = [
                probability(value, "director.auditor_beliefs[]")
                for value in raw_auditor_beliefs
            ]
        belief = sum(auditor_beliefs) / float(len(auditor_beliefs))

        reported_leakage = probability(
            raw.get("singleton_leakage"), "director.singleton_leakage"
        )
        raw_modality_loads = raw.get("modality_loads")
        if raw_modality_loads is None:
            modality_loads: Dict[str, float] = {}
        else:
            if not isinstance(raw_modality_loads, Mapping):
                raise MontageError("director.modality_loads must be an object")
            modality_loads = {
                str(name): probability(value, "director.modality_loads.%s" % name)
                for name, value in raw_modality_loads.items()
            }
        if stage == "screenwriter":
            modality_loads = {"v": 0.0, "t": 0.0, "vt": 0.0}
        leakage = (
            max(modality_loads.values())
            if modality_loads
            else reported_leakage
        )
        raw_revise_owner = str(raw.get("revise_owner", "")).strip().lower()
        owner_token = raw_revise_owner.replace("-", "_").replace(" ", "_")
        owner_aliases = {
            "": "",
            "none": "",
            "no_revision": "",
            "screenwriter": "screenwriter",
            "screen_writer": "screenwriter",
            "visual": "visual_editor",
            "visualeditor": "visual_editor",
            "visual_editor": "visual_editor",
            "caption": "caption_editor",
            "captioneditor": "caption_editor",
            "caption_editor": "caption_editor",
        }
        reported_revise_owner = owner_aliases.get(owner_token, "")
        if owner_token and owner_token not in owner_aliases:
            self._emit(
                "director_revise_owner_ignored",
                stage=stage,
                reported_revise_owner=raw_revise_owner,
            )
        revise_owner = reported_revise_owner
        overloaded = [
            name
            for name, value in modality_loads.items()
            if value > dataset.settings.epsilon
        ]
        if overloaded:
            dominant_modality = max(
                overloaded, key=lambda name: modality_loads[name]
            )
            revise_owner = (
                "visual_editor"
                if dominant_modality == "v"
                else "caption_editor"
            )
        if not local:
            expected_action = "REJECT"
        elif belief > dataset.settings.tau and leakage <= dataset.settings.epsilon:
            expected_action = "ACCEPT"
        else:
            expected_action = "REVISE"
        if expected_action == "REVISE":
            if not overloaded:
                revise_owner = stage if stage in (
                    "screenwriter",
                    "visual_editor",
                    "caption_editor",
                ) else ""
        else:
            revise_owner = ""
        model_action = str(raw.get("action", "")).strip().upper()
        action = expected_action
        if model_action != expected_action:
            self._emit(
                "director_action_normalized",
                stage=stage,
                model_action=model_action or "<empty>",
                expected_action=expected_action,
                target_belief=belief,
                singleton_leakage=leakage,
                revise_owner=revise_owner or "none",
            )
        critique_value = str(raw.get("critique") or "").strip()
        if not critique_value:
            critique_value = "No additional gate critique was provided."
        return DirectorDecision(
            action=action,
            local_compliance=local,
            target_belief=belief,
            singleton_leakage=leakage,
            violations=violations,
            critique=required_english_text(critique_value, "director.critique"),
            agent_id=self.prompt_agent.agent_id,
            model=self.prompt_agent.model_name,
            auditor_beliefs=auditor_beliefs,
            modality_loads=modality_loads,
            lc_components=lc_components,
            revise_owner=revise_owner,
            spurious_edges=spurious_edges,
        )

    @staticmethod
    def _decision_rank(
        decision: DirectorDecision,
    ) -> Tuple[int, int, float, float]:
        action_rank = {"REJECT": 0, "REVISE": 1, "ACCEPT": 2}[
            decision.action
        ]
        return (
            action_rank,
            1 if decision.local_compliance else 0,
            decision.target_belief,
            -decision.singleton_leakage,
        )

    def _log_debate(
        self,
        phase: str,
        round_number: int,
        draft: Dict[str, Any],
        decision: DirectorDecision,
        candidate_number: int = 0,
    ) -> None:
        row = {
            "phase": phase,
            "round": round_number,
            "draft": draft,
            "director": decision.to_dict(),
        }
        if candidate_number:
            row["candidate"] = candidate_number
        self.debate_history.append(row)

    @staticmethod
    def _print_decision(decision: DirectorDecision) -> None:
        print(
            "DIRECTOR: %s | locally_truthful=%s | target_belief=%.3f | "
            "singleton_leakage=%.3f"
            % (
                decision.action,
                str(decision.local_compliance).lower(),
                decision.target_belief,
                decision.singleton_leakage,
            )
        )
        if decision.violations:
            print("  Violations: %s" % "; ".join(decision.violations))
        if decision.modality_loads:
            print(
                "  Modality loads: %s"
                % json.dumps(decision.modality_loads, sort_keys=True)
            )
        if decision.revise_owner:
            print("  Revise owner: %s" % decision.revise_owner)
        print("  Critique: %s\n" % decision.critique)

    @staticmethod
    def _build_publication_plan(
        dataset: MontageDataset, caption_sequence: CaptionSequence
    ) -> List[Dict[str, Any]]:
        try:
            start = datetime.fromisoformat(
                dataset.settings.start_time.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise MontageError("Invalid settings.start_time") from exc
        result = []
        for position, item in enumerate(caption_sequence.units, start=1):
            result.append(
                {
                    "position": position,
                    "unit_id": item["unit_id"],
                    "modality": item["modality"],
                    "caption": item["caption"] or None,
                    "account_id": dataset.settings.account_ids[
                        (position - 1) % len(dataset.settings.account_ids)
                    ],
                    "timestamp": (
                        start
                        + timedelta(
                            minutes=(position - 1)
                            * dataset.settings.interval_minutes
                        )
                    ).isoformat(),
                }
            )
        return result
