"""Provider-neutral conversation interface used by every agent.

Each backend keeps the conversation in its provider's native message format
(so, for Claude, thinking blocks are passed back unchanged) and returns a
normalized ``AssistantTurn``. Agents never touch provider SDKs directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __iadd__(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        return self

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


# Normalized stop reasons
END_TURN = "end_turn"
TOOL_USE = "tool_use"
MAX_TOKENS = "max_tokens"
REFUSAL = "refusal"


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    usage: Usage
    model: str
    request_id: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


class GatewayError(RuntimeError):
    """A model call failed after the SDK's own retries. ``retryable`` says whether a later retry may help."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class Conversation(ABC):
    model: str

    @abstractmethod
    def send_user(self, text: str) -> AssistantTurn: ...

    @abstractmethod
    def send_tool_results(self, results: list[ToolResult]) -> AssistantTurn: ...


class Backend(ABC):
    @abstractmethod
    def start(
        self,
        *,
        system: str,
        tools: list[ToolSpec],
        model: str,
        effort: str | None = None,
        max_tokens: int = 16000,
    ) -> Conversation: ...
