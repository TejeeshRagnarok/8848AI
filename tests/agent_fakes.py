"""Scripted stand-in for the LLM provider.  No network, no API key, no cost."""

import copy
import itertools
import json
from typing import Any

from app.agent.llm import AssistantTurn, ToolCall

_ids = itertools.count(1)


def say(text: str) -> AssistantTurn:
    return AssistantTurn(content=text)


def call(name: str, **arguments: Any) -> AssistantTurn:
    return calls((name, arguments))


def calls(*pairs: tuple[str, Any]) -> AssistantTurn:
    """One model turn requesting several tools.  A ``str`` argument payload is
    passed through verbatim (to simulate malformed JSON)."""
    return AssistantTurn(
        content=None,
        tool_calls=[
            ToolCall(
                id=f"call_{next(_ids)}",
                name=name,
                arguments=args if isinstance(args, str) else json.dumps(args),
            )
            for name, args in pairs
        ],
    )


def last_tool_result(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return json.loads(next(m for m in reversed(messages) if m["role"] == "tool")["content"])


class ScriptedProvider:
    """Returns scripted turns in order.  A step may be an ``AssistantTurn``, an
    exception to raise, or a callable ``messages -> AssistantTurn`` (to build a
    reply from the real tool results the agent sent)."""

    def __init__(self, *steps: Any) -> None:
        self.steps = list(steps)
        self.requests: list[dict[str, Any]] = []

    def complete(self, messages, tools) -> AssistantTurn:
        self.requests.append({"messages": copy.deepcopy(messages), "tools": tools})
        if not self.steps:
            raise AssertionError("unexpected extra model call")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(messages) if callable(step) else step


class RepeatingProvider:
    """Returns the same turn forever (to exercise loop limits)."""

    def __init__(self, turn: AssistantTurn) -> None:
        self.turn = turn
        self.count = 0

    def complete(self, messages, tools) -> AssistantTurn:
        self.count += 1
        return self.turn
