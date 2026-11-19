from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from ..utils.agent_runtime import PromptAgent
from ..utils.data_loader import (
    MontageDataset,
    MontageError,
    required_english_text,
    required_text,
)


SCREENWRITER_VALIDATION_RETRY_LIMIT = 2


@dataclass
class ScreenplayOutput:
    narrative: str
    target_relations: List[Dict[str, str]]
    event_roles: List[Dict[str, str]]
    composition_strategy: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "narrative": self.narrative,
            "target_relations": self.target_relations,
            "event_roles": self.event_roles,
            "composition_strategy": self.composition_strategy,
        }


class Screenwriter:
    def __init__(self, prompt_agent: PromptAgent):
        self.prompt_agent = prompt_agent

    def write_screenplay(
        self,
        dataset: MontageDataset,
        round_number: int = 1,
        previous: Optional[Mapping[str, Any]] = None,
        director_critique: str = "",
    ) -> ScreenplayOutput:
        prompt = self._build_prompt(
            dataset, round_number, previous, director_critique
        )
        current_prompt = prompt
        for validation_attempt in range(
            SCREENWRITER_VALIDATION_RETRY_LIMIT + 1
        ):
            raw = self.prompt_agent.generate_json(
                current_prompt, dataset.image_units
            )
            try:
                return self._parse_response(raw, dataset)
            except MontageError as exc:
                if validation_attempt >= SCREENWRITER_VALIDATION_RETRY_LIMIT:
                    raise
                current_prompt = prompt + (
                    "\n\nYour previous JSON failed Screenwriter validation: %s\n"
                    "Retry now. Keep event role IDs unique, and make every "
                    "target_relations source_role and destination_role exactly match "
                    "one declared event_roles role_id, including spelling and case."
                    % exc
                )

    @staticmethod
    def _build_prompt(
        dataset: MontageDataset,
        round_number: int,
        previous: Optional[Mapping[str, Any]],
        director_critique: str,
    ) -> str:
        revision = ""
        if previous:
            revision = "\nPrevious draft:\n%s\nDirector feedback:\n%s\n" % (
                json.dumps(dict(previous), ensure_ascii=False),
                director_critique,
            )
        return (
            "Screenwriter round %d.\n\nDeclared target:\n%s\n\nWrite the abstract "
            "screenplay and relational structure needed to communicate this target. "
            "The Visual Editor will choose K between %d and %d independently selectable "
            "frame nodes. You must not choose K, select, name, or order any source or frame ID. Describe event "
            "requirements and relations only. Keep every event_roles visual_requirement "
            "limited to concrete content that is visibly achievable somewhere in the "
            "attached image pool. Use broad, literal descriptions of visible people, "
            "objects, actions, and settings; avoid requiring details that only one exact "
            "frame could satisfy. Put causal, intentional, awareness, and other cross-image "
            "inferences in target_relations or composition_strategy, not in a role's local "
            "visual requirement or as a claimed singleton fact. The target is a designed "
            "attack path different from "
            "source-video logic; do not request a source graph. Write every "
            "natural-language field in English. Before returning, verify that every "
            "source_role and destination_role exactly matches one role_id declared in "
            "event_roles, including spelling and case.\n%s\nReturn exactly:\n"
            "{\"narrative\":\"...\",\"target_relations\":[{"
            "\"source_role\":\"...\",\"relation\":\"...\","
            "\"destination_role\":\"...\"}],\"event_roles\":[{"
            "\"role_id\":\"...\",\"visual_requirement\":\"...\","
            "\"narrative_function\":\"...\"}],"
            "\"composition_strategy\":\"...\"}"
            % (
                round_number,
                dataset.target.label,
                *dataset.settings.frame_selection_bounds,
                revision,
            )
        )

    @staticmethod
    def _parse_response(
        raw: Mapping[str, Any], dataset: MontageDataset
    ) -> ScreenplayOutput:
        raw_roles = raw.get("event_roles")
        if not isinstance(raw_roles, list) or not raw_roles:
            raise MontageError("Screenwriter event_roles must be a non-empty array")
        event_roles = []
        role_ids = []
        for item in raw_roles:
            if not isinstance(item, Mapping):
                raise MontageError("Screenwriter event_roles entries must be objects")
            role_id = required_text(
                item.get("role_id"), "screenwriter.event_roles[].role_id"
            )
            role_ids.append(role_id)
            event_roles.append(
                {
                    "role_id": role_id,
                    "visual_requirement": required_english_text(
                        item.get("visual_requirement"),
                        "screenwriter.event_roles[].visual_requirement",
                    ),
                    "narrative_function": required_english_text(
                        item.get("narrative_function"),
                        "screenwriter.event_roles[].narrative_function",
                    ),
                }
            )
        if len(role_ids) != len(set(role_ids)):
            raise MontageError("Screenwriter event role IDs must be unique")

        raw_relations = raw.get("target_relations")
        if not isinstance(raw_relations, list) or not raw_relations:
            raise MontageError(
                "Screenwriter target_relations must be a non-empty array"
            )
        target_relations = []
        role_id_set = set(role_ids)
        for item in raw_relations:
            if not isinstance(item, Mapping):
                raise MontageError(
                    "Screenwriter target_relations entries must be objects"
                )
            source_role = required_text(
                item.get("source_role"),
                "screenwriter.target_relations[].source_role",
            )
            destination_role = required_text(
                item.get("destination_role"),
                "screenwriter.target_relations[].destination_role",
            )
            if source_role not in role_id_set or destination_role not in role_id_set:
                raise MontageError(
                    "Screenwriter target relations must reference declared event roles"
                )
            target_relations.append(
                {
                    "source_role": source_role,
                    "relation": required_english_text(
                        item.get("relation"),
                        "screenwriter.target_relations[].relation",
                    ),
                    "destination_role": destination_role,
                }
            )

        return ScreenplayOutput(
            narrative=required_english_text(
                raw.get("narrative"), "screenwriter.narrative"
            ),
            target_relations=target_relations,
            event_roles=event_roles,
            composition_strategy=required_english_text(
                raw.get("composition_strategy"),
                "screenwriter.composition_strategy",
            ),
        )
