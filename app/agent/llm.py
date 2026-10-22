"""LLM provider abstraction: OpenAI-compatible chat completions with tool calling.

The rest of the agent depends only on ``ChatProvider`` and the small result
types below, so providers can be swapped (or faked in tests) freely.

Errors are reduced to three safe categories.  Their messages are fixed
strings: raw provider responses, URLs and keys never reach callers or logs.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from app.config import Settings, settings

logger = logging.getLogger(__name__)


# ── Errors ───────────────────────────────────────────────────────────────────


class LLMError(Exception):
    """Base class; ``str(exc)`` is always safe to show."""


class LLMNotConfiguredError(LLMError):
    def __init__(self) -> None:
        super().__init__("The ordering assistant is not configured.")


class LLMUnavailableError(LLMError):
    """Timeouts, outages, rate limits, or rejected credentials."""

    def __init__(self, reason: str) -> None:
        super().__init__("The ordering assistant is temporarily unavailable.")
        self.reason = reason  # short internal category, safe to log


class LLMResponseError(LLMError):
    """The provider answered, but not in the expected shape."""

    def __init__(self, reason: str) -> None:
        super().__init__("The ordering assistant returned an invalid response.")
        self.reason = reason


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text from the model; validated before use


@dataclass(frozen=True)
class AssistantTurn:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> dict[str, Any]:
        """This turn as an OpenAI-format assistant message (for history)."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return message


class ChatProvider(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AssistantTurn:
        """Run one chat-completion round.  Raises ``LLMError`` subclasses only."""
        ...


# ── OpenAI-compatible implementation ─────────────────────────────────────────


def parse_completion(payload: Any) -> AssistantTurn:
    """Strictly parse a chat-completions response body."""
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise LLMResponseError("missing choices[0].message") from None
    if not isinstance(message, dict):
        raise LLMResponseError("message is not an object")

    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise LLMResponseError("content is not a string")

    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise LLMResponseError("tool_calls is not a list")
    calls = []
    for raw in raw_calls:
        try:
            function = raw["function"]
            call = ToolCall(id=raw["id"], name=function["name"], arguments=function.get("arguments") or "{}")
        except (KeyError, TypeError):
            raise LLMResponseError("malformed tool call") from None
        if raw.get("type", "function") != "function" or not all(
            isinstance(v, str) for v in (call.id, call.name, call.arguments)
        ):
            raise LLMResponseError("malformed tool call")
        calls.append(call)

    if not calls and not (content and content.strip()):
        raise LLMResponseError("empty response")
    return AssistantTurn(content=content, tool_calls=calls)


class OpenAICompatibleProvider:
    """Calls ``POST {base_url}/chat/completions``."""

    def __init__(
        self, config: Settings, client: httpx.Client | None = None, sleep=time.sleep
    ) -> None:
        if config.llm_api_key is None or not config.llm_model:
            raise LLMNotConfiguredError()
        self._max_rate_wait = config.llm_rate_limit_max_wait_seconds
        self._sleep = sleep
        self._model = config.llm_model
        self._max_tokens = config.llm_max_tokens
        self._temperature = config.llm_temperature
        self._url = config.llm_base_url.rstrip("/") + "/chat/completions"
        self._host = urlsplit(self._url).hostname or "unknown"  # the only part ever logged
        self._client = client or httpx.Client(timeout=httpx.Timeout(config.llm_timeout_seconds))
        self._headers = {"Authorization": f"Bearer {config.llm_api_key.get_secret_value()}"}

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AssistantTurn:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            body["temperature"] = self._temperature

        response = self._post(body)
        wait = _retry_after(response)
        if response.status_code == 429 and wait is not None and wait <= self._max_rate_wait:
            # A short rate-limit pause (e.g. a per-minute token budget) is
            # better waited out once than turned into a failed turn or call.
            self._log("rate_limited_retrying", 429)
            self._sleep(wait)
            response = self._post(body)

        status = response.status_code
        if status in (401, 403):
            self._log("authentication_failed", status)
            raise LLMUnavailableError("authentication_failed")
        if status == 429:
            self._log("rate_limited", status)
            raise LLMUnavailableError("rate_limited")
        if status >= 500:
            self._log("provider_error", status)
            raise LLMUnavailableError("provider_error")
        if status != 200:
            self._log("unexpected_status", status)
            raise LLMResponseError("unexpected_status")

        try:
            payload = response.json()
        except ValueError:
            self._log("invalid_json", status)
            raise LLMResponseError("invalid_json") from None
        return parse_completion(payload)

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        try:
            return self._client.post(self._url, json=body, headers=self._headers)
        except httpx.TimeoutException:
            self._log("timeout")
            raise LLMUnavailableError("timeout") from None
        except httpx.HTTPError:
            self._log("connection_error")
            raise LLMUnavailableError("connection_error") from None

    def _log(self, reason: str, status: int | None = None) -> None:
        logger.warning("LLM request to %s failed: %s (status %s)", self._host, reason, status)


def _retry_after(response: httpx.Response) -> float | None:
    """Seconds from a numeric ``Retry-After`` header, if present."""
    try:
        return max(0.0, float(response.headers["retry-after"]))
    except (KeyError, ValueError):
        return None


_provider: OpenAICompatibleProvider | None = None


def get_provider() -> ChatProvider:
    """Return the configured provider (reusing one HTTP client).

    Raises ``LLMNotConfiguredError`` when no API key or model is set.
    """
    global _provider  # noqa: PLW0603
    if _provider is None:
        _provider = OpenAICompatibleProvider(settings)
    return _provider
