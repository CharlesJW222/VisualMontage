from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from ..utils.agent_runtime import PromptAgent
from ..utils.data_loader import (
    MontageDataset,
    MontageError,
)
from .screenwriter import ScreenplayOutput


VISUAL_VALIDATION_RETRY_LIMIT = 2


@dataclass
class VisualSequence:
    ordered_unit_ids: List[str]
    sequencing_strategy: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ordered_unit_ids": self.ordered_unit_ids,
            "sequencing_strategy": self.sequencing_strategy,
        }


class VisualEditor:
    def __init__(self, prompt_agent: PromptAgent):
        self.prompt_agent = prompt_agent

    def create_sequence(
        self,
        dataset: MontageDataset,
        screenplay: ScreenplayOutput,
        round_number: int = 1,
        previous: Optional[Mapping[str, Any]] = None,
        director_critique: str = "",
    ) -> VisualSequence:
        minimum, maximum = dataset.settings.frame_selection_bounds
        if len(dataset.image_units) < minimum:
            raise MontageError(
                "Candidate Frame Pool contains fewer frames than the configured "
                "minimum frame selection count (%d < %d)"
                % (len(dataset.image_units), minimum)
            )
        prompt = self._build_prompt(
            dataset, screenplay, round_number, previous, director_critique
        )
        current_prompt = prompt
        for validation_attempt in range(VISUAL_VALIDATION_RETRY_LIMIT + 1):
            raw = self.prompt_agent.generate_json(
                current_prompt, dataset.image_units
            )
            try:
                return self._parse_response(raw, dataset)
            except MontageError as exc:
                if validation_attempt >= VISUAL_VALIDATION_RETRY_LIMIT:
                    raise
                current_prompt = prompt + (
                    "\n\nYour previous JSON failed Visual Editor validation: %s\n"
                    "Retry now. Choose K with %d <= K <= %d distinct supplied frame IDs in "
                    "ordered_unit_ids. Return the requested VisualSequence schema."
                    % (
                        exc,
                        minimum,
                        maximum,
                    )
                )

    @staticmethod
    def _build_prompt(
        dataset: MontageDataset,
        screenplay: ScreenplayOutput,
        round_number: int,
        previous: Optional[Mapping[str, Any]],
        director_critique: str,
    ) -> str:
        revision = ""
        if previous:
            revision = (
                "\n"
                "\n"
                "Previous proposal:\n"
                "%s\n"
                "Director feedback:\n"
                "%s\n"
                % (json.dumps(dict(previous), ensure_ascii=False), director_critique)
            )
        elif director_critique:
            revision = "\n\nDirector feedback:\n%s" % director_critique
        frame_pool = [{"frame_id": unit.unit_id} for unit in dataset.image_units]
        minimum, maximum = dataset.settings.frame_selection_bounds
        length_instruction = (
            "Select exactly %d distinct supplied frame IDs" % minimum
            if minimum == maximum else
            "Choose K yourself within %d <= K <= %d and select K distinct supplied frame IDs"
            % (minimum, maximum)
        )
        return (
            "Visual Editor round %d.\n"
            "\n"
            "Declared target:\n"
            "%s\n"
            "\n"
            "Approved Screenwriter screenplay:\n"
            "%s\n"
            "\n"
            "Complete Candidate Frame Pool:\n"
            "%s\n"
            "\n"
            "Selection constraint: %s.\n"
            "Heuristic parameters: alpha=%s, beta=%s, tau=%s.\n"
            "%s\n"
            "\n"
            "Return exactly:\n"
            "{\"ordered_unit_ids\":[\"...\"],\"sequencing_strategy\":\"...\"}"
            % (
                round_number,
                dataset.target.label,
                json.dumps(screenplay.to_dict(), indent=2, ensure_ascii=False),
                json.dumps(frame_pool, indent=2, ensure_ascii=False),
                length_instruction,
                dataset.settings.search_alpha,
                dataset.settings.search_beta,
                dataset.settings.tau,
                revision,
            )
        )

    @staticmethod
    def _parse_response(
        raw: Mapping[str, Any], dataset: MontageDataset
    ) -> VisualSequence:
        if not isinstance(raw, Mapping):
            raise MontageError("Visual Editor response must be a JSON object")
        ordered = raw.get("ordered_unit_ids")
        if not isinstance(ordered, list):
            raise MontageError(
                "Visual Editor ordered_unit_ids must be an array"
            )
        if not all(isinstance(item, str) for item in ordered):
            raise MontageError("Visual Editor frame IDs must be strings")
        if len(ordered) != len(set(ordered)):
            raise MontageError("Visual Editor frame IDs must be distinct")
        lookup = {unit.unit_id: unit for unit in dataset.image_units}
        unknown = [unit_id for unit_id in ordered if unit_id not in lookup]
        if unknown:
            raise MontageError(
                "Visual Editor selected unknown frame IDs: %s" % unknown
            )
        minimum, maximum = dataset.settings.frame_selection_bounds
        if not minimum <= len(ordered) <= maximum:
            expected = (
                "exactly %d" % minimum if minimum == maximum
                else "between %d and %d" % (minimum, maximum)
            )
            raise MontageError(
                "Visual Editor must select %s frame nodes; received %d"
                % (expected, len(ordered))
            )
        strategy = raw.get("sequencing_strategy")
        if not isinstance(strategy, str):
            raise MontageError("Visual Editor sequencing_strategy must be a string")
        return VisualSequence(
            ordered_unit_ids=ordered,
            sequencing_strategy=strategy,
        )
