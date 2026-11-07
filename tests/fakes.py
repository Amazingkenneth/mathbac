"""Scripted stand-ins for model backends, so the harness can be tested without API calls."""

from __future__ import annotations

import itertools
from typing import Callable

from mathbac.gateway import AssistantTurn, Backend, Conversation, ToolCall, ToolResult, ToolSpec, Usage

_ids = itertools.count()


def call(tool_name: str, /, **inp) -> ToolCall:
    return ToolCall(id=f"call_{next(_ids)}", name=tool_name, input=inp)


def turn(*calls: ToolCall, text: str = "") -> AssistantTurn:
    return AssistantTurn(text=text, tool_calls=list(calls), stop_reason="tool_use" if calls else "end_turn",
                         usage=Usage(input_tokens=100, output_tokens=20), model="claude-opus-5-5")


class FakeConversation(Conversation):
    def __init__(self, system: str, tools: list[ToolSpec], policy: "Policy"):
        self.model = "claude-opus-5-5"
        self.system = system
        self.tools = tools
        self.policy = policy
        self.inbox: list[str | list[ToolResult]] = []
        self.step = 0

    def send_user(self, text: str) -> AssistantTurn:
        self.inbox.append(text)
        return self._next()

    def send_tool_results(self, results: list[ToolResult]) -> AssistantTurn:
        self.inbox.append(results)
        return self._next()

    def _next(self) -> AssistantTurn:
        t = self.policy(self)
        self.step += 1
        return t


Policy = Callable[[FakeConversation], AssistantTurn]


class FakeBackend(Backend):
    def __init__(self, policy: Policy):
        self.policy = policy
        self.conversations: list[FakeConversation] = []

    def start(self, *, system, tools, model, effort=None, max_tokens=16000) -> Conversation:
        conv = FakeConversation(system, tools, self.policy)
        self.conversations.append(conv)
        return conv


def scripted(*turns: AssistantTurn) -> Policy:
    """Return the given turns in order, then end the turn without tool calls."""

    def policy(conv: FakeConversation) -> AssistantTurn:
        return turns[conv.step] if conv.step < len(turns) else turn(text="done")

    return policy


def fake_router(policies: dict[str, Policy]) -> tuple[dict, dict[str, Backend]]:
    """Models config and backends that send each role to its own scripted policy."""
    backends = {f"fake_{r}": FakeBackend(p) for r, p in policies.items()}
    cfg = {"defaults": {"provider": "fake_default", "model": "claude-opus-5-5"},
           "roles": {r: {"provider": f"fake_{r}"} for r in policies}}
    backends.setdefault("fake_default", FakeBackend(scripted()))
    return cfg, backends
