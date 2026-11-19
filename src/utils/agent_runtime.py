from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .data_loader import ImageUnit, MontageError
from .json_utils import StructuredResponseError, parse_json_object


def stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AgentCache:
    def __init__(self, path: Optional[str]):
        self.path = Path(path).expanduser().resolve() if path else None
        self.lock = threading.RLock()
        self.data: Dict[str, Any] = {}
        if self.path and self.path.is_file():
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    self.data = value
            except (OSError, json.JSONDecodeError):
                self.data = {}

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        with self.lock:
            self.data[key] = value
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_suffix(self.path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(self.data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                temporary.replace(self.path)


class PromptAgent:
    def __init__(
        self,
        agent_id: str,
        llm_client: Any,
        system_prompt: str,
        cache: AgentCache,
        json_retries: int = 1,
        progress_callback=None,
    ):
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.cache = cache
        self.json_retries = json_retries
        self.progress_callback = progress_callback
        self.requests = 0
        self.cache_hits = 0

    @property
    def model_name(self) -> str:
        return str(getattr(self.llm_client, "model_name", "unknown"))

    def _emit(self, event: str, **details: Any) -> None:
        if self.progress_callback:
            self.progress_callback(
                {"event": event, "agent": self.agent_id, **details}
            )

    def generate_json(
        self,
        prompt: str,
        image_units: Sequence[ImageUnit],
        ordered_unit_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        lookup = {unit.unit_id: unit for unit in image_units}
        order = list(
            ordered_unit_ids or [unit.unit_id for unit in image_units]
        )
        unknown = [unit_id for unit_id in order if unit_id not in lookup]
        if unknown:
            raise MontageError(
                "Agent %s received unknown image units: %s"
                % (self.agent_id, unknown)
            )

        image_paths = []
        descriptors = []
        image_hashes = []
        for position, unit_id in enumerate(order, start=1):
            unit = lookup[unit_id]
            image_hashes.append(unit.image_sha256)
            for frame_index, path in enumerate(unit.image_paths, start=1):
                image_paths.append(path)
                descriptors.append(
                    "frame_id=%s ordered_position=%d image_index=%d"
                    % (unit_id, position, frame_index)
                )

        cache_key = stable_digest(
            {
                "schema": "prompt_agent_request.v2",
                "agent_id": self.agent_id,
                "model": self.model_name,
                "system_prompt": self.system_prompt,
                "prompt": prompt,
                "images": image_hashes,
                "descriptors": descriptors,
            }
        )
        cached = self.cache.get(cache_key)
        if isinstance(cached, dict):
            self.cache_hits += 1
            self._emit("agent_cache_hit", model=self.model_name)
            return dict(cached)

        generate = getattr(self.llm_client, "generate_with_images", None)
        if generate is None:
            raise MontageError(
                "Agent %s requires a vision-capable client" % self.agent_id
            )

        current_prompt = prompt
        last_error = None
        for attempt in range(self.json_retries + 1):
            self.requests += 1
            started = time.monotonic()
            self._emit(
                "agent_request_start", model=self.model_name, attempt=attempt + 1
            )
            response = generate(
                current_prompt,
                image_paths,
                system_prompt=self.system_prompt,
                image_descriptors=descriptors,
            )
            try:
                value = parse_json_object(response)
                self.cache.set(cache_key, value)
                self._emit(
                    "agent_request_complete",
                    model=self.model_name,
                    elapsed_seconds=round(time.monotonic() - started, 6),
                )
                return value
            except StructuredResponseError as exc:
                last_error = exc
                current_prompt = prompt + (
                    "\n\nYour previous answer was invalid. Return one complete JSON "
                    "object only, with no Markdown fences or explanation."
                )
        raise MontageError(
            "%s returned invalid JSON: %s" % (self.agent_id, last_error)
        )

    def stats(self) -> Dict[str, Any]:
        client_stats = getattr(self.llm_client, "performance_stats", None)
        return {
            "agent_id": self.agent_id,
            "model": self.model_name,
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "client": client_stats() if client_stats else {},
        }
