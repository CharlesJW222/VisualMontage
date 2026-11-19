import base64
import json
import mimetypes
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

import requests


class LLMClientError(RuntimeError):
    pass
def is_official_openai_base_url(base_url: Any) -> bool:
    try:
        host = (urlparse(str(base_url or "")).hostname or "").lower()
    except ValueError:
        return False
    return host == "api.openai.com"


def is_reasoning_model(model_name: Any) -> bool:
    normalized = str(model_name or "").strip().lower()
    return normalized.startswith(("gpt-5", "gpt-6"))


def default_max_tokens_field(model_name: Any) -> str:
    if is_reasoning_model(model_name):
        return "max_completion_tokens"
    return "max_tokens"


class BaseLLMClient(ABC):
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 2000):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_with_history(self, messages: List[Dict[str, Any]]) -> str:
        raise NotImplementedError

    def generate_multimodal(
        self,
        content: Sequence[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(BaseLLMClient):
    RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        api_key_header: str = "Authorization",
        api_key_prefix: str = "Bearer ",
        endpoint_path: str = "chat/completions",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        max_tokens_field: str = "max_tokens",
        include_temperature: bool = True,
        timeout_seconds: int = 180,
        connect_timeout_seconds: int = 20,
        max_retries: int = 3,
        verify_ssl: bool = True,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_name, temperature, max_tokens)
        if not base_url:
            raise ValueError("Private API base_url is required")
        if not model_name:
            raise ValueError("Private API model name is required")

        self.endpoint = self._build_endpoint(base_url, endpoint_path)
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.api_key_prefix = api_key_prefix
        self.max_tokens_field = max_tokens_field
        self.include_temperature = include_temperature
        self.timeout_seconds = timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl
        self.extra_body = extra_body or {}
        self._session_headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if api_key and api_key_header:
            self._session_headers[api_key_header] = "%s%s" % (api_key_prefix, api_key)
        if extra_headers:
            self._session_headers.update(extra_headers)

        self.session = self._new_session()
        self._owner_thread_id = threading.get_ident()
        self._thread_state = threading.local()

        self._image_cache: Dict[str, tuple] = {}
        self._image_cache_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = {
            "logical_requests": 0,
            "http_attempts": 0,
            "retries": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_request_seconds": 0.0,
            "responses_with_usage": 0,
            "reported_prompt_tokens": 0,
            "reported_completion_tokens": 0,
            "reported_total_tokens": 0,
            "image_cache_hits": 0,
            "image_cache_misses": 0,
        }

    def _new_session(self):
        session = requests.Session()
        session.headers.update(self._session_headers)
        return session

    def _session_for_current_thread(self):
        if threading.get_ident() == self._owner_thread_id:
            return self.session
        session = getattr(self._thread_state, "session", None)
        if session is None:
            session = self._new_session()
            self._thread_state.session = session
        return session

    @staticmethod
    def _build_endpoint(base_url: str, endpoint_path: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        endpoint = endpoint_path.strip("/")
        return "%s/%s" % (base, endpoint) if endpoint else base

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.generate_with_history(messages)

    def generate_with_history(self, messages: List[Dict[str, Any]]) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            self.max_tokens_field: self.max_tokens,
        }
        if self.include_temperature:
            payload["temperature"] = self.temperature
        payload.update(self.extra_body)
        data = self._post_json(payload)
        return self._extract_text(data)

    def generate_with_images(
        self,
        prompt: str,
        image_paths: Sequence[str],
        system_prompt: Optional[str] = None,
        image_descriptors: Optional[Sequence[str]] = None,
    ) -> str:
        if not image_paths:
            raise ValueError("At least one image is required for a multimodal request")
        if image_descriptors is not None and len(image_descriptors) != len(image_paths):
            raise ValueError("image_descriptors must match image_paths in length")

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for index, image_path in enumerate(image_paths):
            if image_descriptors is not None:
                content.append({"type": "text", "text": str(image_descriptors[index])})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(image_path)},
                }
            )

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return self.generate_with_history(messages)

    def generate_multimodal(
        self,
        content: Sequence[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> str:
        if not content:
            raise ValueError("Multimodal content must not be empty")

        request_content: List[Dict[str, Any]] = []
        for index, block in enumerate(content):
            block_type = str(block.get("type") or "").strip()
            if block_type == "text":
                text = str(block.get("text") or "")
                if not text:
                    raise ValueError(
                        "Multimodal text block %d must not be empty" % index
                    )
                request_content.append({"type": "text", "text": text})
            elif block_type == "image_path":
                image_path = str(block.get("path") or "").strip()
                if not image_path:
                    raise ValueError(
                        "Multimodal image block %d requires path" % index
                    )
                request_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_data_url(image_path)},
                    }
                )
            else:
                raise ValueError(
                    "Unsupported multimodal block type at index %d: %s"
                    % (index, block_type or "<empty>")
                )

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": request_content})
        return self.generate_with_history(messages)

    def _image_data_url(self, image_path: str) -> str:
        path = Path(image_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise LLMClientError("Image file not found: %s" % path)
        stat = path.stat()
        cache_key = str(path)
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        with self._image_cache_lock:
            cached = self._image_cache.get(cache_key)
            if cached and cached[:3] == signature:
                with self._stats_lock:
                    self._stats["image_cache_hits"] += 1
                return cached[3]
            mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            data_url = "data:%s;base64,%s" % (mime_type, encoded)
            self._image_cache[cache_key] = (
                signature[0],
                signature[1],
                signature[2],
                data_url,
            )
            with self._stats_lock:
                self._stats["image_cache_misses"] += 1
            return data_url

    def clear_image_cache(self) -> None:
        with self._image_cache_lock:
            self._image_cache.clear()

    def performance_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            result = dict(self._stats)
        result["total_request_seconds"] = round(
            float(result["total_request_seconds"]), 6
        )
        result.update({
            "model": self.model_name,
            "endpoint": self.endpoint,
            "include_temperature": self.include_temperature,
            "timeout_seconds": self.timeout_seconds,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "max_retries": self.max_retries,
            "maximum_attempts_per_request": self.max_retries + 1,
        })
        return result

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        started = time.monotonic()
        succeeded = False
        with self._stats_lock:
            self._stats["logical_requests"] += 1
        last_error = None
        encoded_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            for attempt in range(self.max_retries + 1):
                with self._stats_lock:
                    self._stats["http_attempts"] += 1
                try:
                    response = self._session_for_current_thread().post(
                        self.endpoint,
                        data=encoded_payload,
                        timeout=(self.connect_timeout_seconds, self.timeout_seconds),
                        verify=self.verify_ssl,
                    )
                    if response.status_code >= 400:
                        message = "Private API returned HTTP %s: %s" % (
                            response.status_code,
                            response.text[:1000],
                        )
                        if response.status_code not in self.RETRYABLE_STATUS_CODES:
                            raise LLMClientError(message)
                        last_error = LLMClientError(message)
                    else:
                        try:
                            parsed = response.json()
                        except ValueError as exc:
                            raise LLMClientError(
                                "Private API returned non-JSON data: %s"
                                % response.text[:1000]
                            ) from exc
                        if not isinstance(parsed, dict):
                            raise LLMClientError(
                                "Private API JSON response must be an object"
                            )
                        self._record_reported_usage(parsed)
                        succeeded = True
                        return parsed
                except requests.Timeout as exc:
                    last_error = LLMClientError(
                        "Private API timeout on attempt %d/%d "
                        "(connect=%ss, read=%ss): %s"
                        % (
                            attempt + 1,
                            self.max_retries + 1,
                            self.connect_timeout_seconds,
                            self.timeout_seconds,
                            exc,
                        )
                    )
                except requests.ConnectionError as exc:
                    last_error = LLMClientError(
                        "Private API connection failed on attempt %d/%d: %s"
                        % (attempt + 1, self.max_retries + 1, exc)
                    )

                if attempt < self.max_retries:
                    with self._stats_lock:
                        self._stats["retries"] += 1
                    time.sleep(min(2 ** attempt, 8))

            raise last_error or LLMClientError("Private API request failed")
        finally:
            with self._stats_lock:
                status_key = "successful_requests" if succeeded else "failed_requests"
                self._stats[status_key] += 1
                self._stats["total_request_seconds"] += time.monotonic() - started

    def _record_reported_usage(self, response: Dict[str, Any]) -> None:
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return

        def token_count(*names: str) -> int:
            for name in names:
                value = usage.get(name)
                if isinstance(value, bool):
                    continue
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    continue
                if number >= 0:
                    return number
            return 0

        prompt_tokens = token_count("prompt_tokens", "input_tokens")
        completion_tokens = token_count("completion_tokens", "output_tokens")
        total_tokens = token_count("total_tokens")
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens
        with self._stats_lock:
            self._stats["responses_with_usage"] += 1
            self._stats["reported_prompt_tokens"] += prompt_tokens
            self._stats["reported_completion_tokens"] += completion_tokens
            self._stats["reported_total_tokens"] += total_tokens

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        try:
            choice = data["choices"][0]
            if isinstance(choice.get("message"), dict):
                content = choice["message"].get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    text_parts = [
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") in (None, "text")
                    ]
                    combined = "".join(text_parts).strip()
                    if combined:
                        return combined
            if isinstance(choice.get("text"), str):
                return choice["text"]
        except (KeyError, IndexError, TypeError):
            pass

        for key in ("output_text", "response", "text"):
            if isinstance(data.get(key), str):
                return data[key]

        raise LLMClientError(
            "Cannot find generated text in private API response. Top-level keys: %s"
            % sorted(data.keys())
        )


def create_private_client(config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None):
    values = dict(config)
    for key, value in (overrides or {}).items():
        if value is not None:
            values[key] = value

    base_url = values.get("base_url", "")
    official_openai = is_official_openai_base_url(base_url)
    key_env = values.get("api_key_env", "PRIVATE_API_KEY")
    api_key = os.getenv(key_env) if key_env else None
    if not api_key and official_openai and key_env != "OPENAI_API_KEY":
        api_key = os.getenv("OPENAI_API_KEY")
    if key_env and not api_key:
        expected = [str(key_env)]
        if official_openai and key_env != "OPENAI_API_KEY":
            expected.append("OPENAI_API_KEY")
        raise ValueError(
            "API key environment variable is not set; expected one of: %s"
            % ", ".join(expected)
        )

    include_temperature = values.get("include_temperature")
    if include_temperature is None:
        include_temperature = not (
            official_openai and is_reasoning_model(values.get("model"))
        )

    return OpenAICompatibleClient(
        base_url=base_url,
        endpoint_path=values.get("endpoint_path", "chat/completions"),
        model_name=values.get("model", ""),
        api_key=api_key,
        api_key_header=values.get("api_key_header", "Authorization"),
        api_key_prefix=values.get("api_key_prefix", "Bearer "),
        temperature=float(values.get("temperature", 0.7)),
        max_tokens=int(values.get("max_tokens", 2000)),
        max_tokens_field=(
            values.get("max_tokens_field")
            or default_max_tokens_field(values.get("model"))
        ),
        include_temperature=bool(include_temperature),
        timeout_seconds=int(values.get("timeout_seconds", 180)),
        connect_timeout_seconds=int(values.get("connect_timeout_seconds", 20)),
        max_retries=int(values.get("max_retries", 3)),
        verify_ssl=bool(values.get("verify_ssl", True)),
        extra_headers=values.get("extra_headers") or {},
        extra_body=values.get("extra_body") or {},
    )
