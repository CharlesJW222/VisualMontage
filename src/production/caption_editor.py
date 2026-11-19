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
from .screenwriter import ScreenplayOutput
from .visual_editor import VisualSequence


CAPTION_LENGTH_RETRY_LIMIT = 2


class CaptionLengthError(MontageError):
    pass
@dataclass
class CaptionSequence:
    units: List[Dict[str, str]]
    caption_strategy: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "units": self.units,
            "caption_strategy": self.caption_strategy,
        }


class CaptionEditor:
    def __init__(self, prompt_agent: PromptAgent):
        self.prompt_agent = prompt_agent

    def create_captions(
        self,
        dataset: MontageDataset,
        screenplay: ScreenplayOutput,
        visual_sequence: VisualSequence,
        round_number: int = 1,
        previous: Optional[Mapping[str, Any]] = None,
        director_critique: str = "",
    ) -> CaptionSequence:
        prompt = self._build_prompt(
            dataset,
            screenplay,
            visual_sequence,
            round_number,
            previous,
            director_critique,
        )
        current_prompt = prompt
        for attempt in range(CAPTION_LENGTH_RETRY_LIMIT + 1):
            raw = self.prompt_agent.generate_json(
                current_prompt, dataset.image_units, visual_sequence.ordered_unit_ids
            )
            try:
                return self._parse_response(raw, visual_sequence)
            except CaptionLengthError:
                if attempt >= CAPTION_LENGTH_RETRY_LIMIT:
                    raise
                current_prompt = prompt + (
                    "\n\nCaption length repair attempt %d: your previous JSON contains "
                    "a caption longer than 18 words. Shorten every overlong caption "
                    "to at most 18 whitespace-separated words while preserving its "
                    "locally grounded meaning. Preserve frame IDs, order, modalities, "
                    "and already valid captions. Return the complete corrected JSON."
                    "\nPrevious JSON:\n%s"
                    % (attempt + 1, json.dumps(raw, ensure_ascii=False))
                )

    @staticmethod
    def _build_prompt(
        dataset: MontageDataset,
        screenplay: ScreenplayOutput,
        visual_sequence: VisualSequence,
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
            "Caption Editor round %d.\n\nDeclared target:\n%s\n\nApproved "
            "Screenwriter draft:\n%s\n\nApproved Visual Editor draft:\n%s\n\n"
            "Preserve ordered_unit_ids exactly. Each ID represents one frame. For each "
            "frame choose exactly one modality: v publishes the image with an empty "
            "caption; t publishes a non-empty caption while withholding the associated "
            "image; vt publishes the image with a non-empty caption. For t, use the "
            "attached source frame only as private grounding evidence. Every t or vt "
            "caption must remain locally faithful to that one associated frame."
            "\n%s\nReturn exactly:\n"
            "{\"units\":[{\"unit_id\":\"...\",\"modality\":\"v, t, or vt\","
            "\"caption\":\"\"}],\"caption_strategy\":\"...\"}"
            % (
                round_number,
                dataset.target.label,
                json.dumps(screenplay.to_dict(), indent=2, ensure_ascii=False),
                json.dumps(
                    visual_sequence.to_dict(), indent=2, ensure_ascii=False
                ),
                revision,
            )
        )

    @staticmethod
    def _parse_response(
        raw: Mapping[str, Any], visual_sequence: VisualSequence
    ) -> CaptionSequence:
        raw_units = raw.get("units")
        ordered_ids = visual_sequence.ordered_unit_ids
        if not isinstance(raw_units, list) or len(raw_units) != len(ordered_ids):
            raise MontageError(
                "Caption Editor frames must match the Visual Editor order length"
            )

        units = []
        unit_ids = []
        for index, item in enumerate(raw_units):
            if not isinstance(item, Mapping):
                raise MontageError(
                    "Caption Editor frame entries must be objects"
                )
            unit_id = CaptionEditor._normalise_unit_id(item, index)
            modality = str(item.get("modality", "")).strip().lower()
            caption = str(item.get("caption", "")).strip()
            if modality not in ("v", "t", "vt"):
                raise MontageError("Caption Editor modality must be v, t, or vt")
            if modality in ("t", "vt") and not caption:
                raise MontageError(
                    "Caption Editor t and vt units require a caption"
                )
            if modality == "v" and caption:
                raise MontageError(
                    "Caption Editor v units must have an empty caption"
                )
            if caption:
                caption = required_english_text(
                    caption, "caption_editor.caption"
                )
                if len(caption.split()) > 18:
                    raise CaptionLengthError(
                        "Caption Editor captions must contain at most 18 words"
                    )
            unit_ids.append(unit_id)
            units.append(
                {"unit_id": unit_id, "modality": modality, "caption": caption}
            )
        if unit_ids != ordered_ids:
            raise MontageError(
                "Caption Editor must preserve the Visual Editor frame order exactly"
            )
        return CaptionSequence(
            units=units,
            caption_strategy=required_english_text(
                raw.get("caption_strategy"), "caption_editor.caption_strategy"
            ),
        )

    @staticmethod
    def _normalise_unit_id(item: Mapping[str, Any], index: int) -> str:
        value = item.get("unit_id")
        if not str(value or "").strip():
            aliases = ("unitid", "unitId", "frame_id", "frameId")
            candidates = [
                item.get(alias)
                for alias in aliases
                if str(item.get(alias) or "").strip()
            ]
            if len(candidates) > 1:
                raise MontageError(
                    "Caption Editor frame %d has ambiguous unit ID fields"
                    % (index + 1)
                )
            if candidates:
                value = candidates[0]
        return required_text(value, "caption_editor.unit_id")
