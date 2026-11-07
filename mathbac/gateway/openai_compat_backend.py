"""Backend for OpenAI-compatible chat endpoints: OpenRouter (any model it serves) and OpenAI directly.

Used for the model-mixing experiment (E3) and for GPT models. Claude models go
through ``anthropic_backend`` instead.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .base import (
    END_TURN,
    MAX_TOKENS,
    REFUSAL,
    TOOL_USE,
    AssistantTurn,
    Backend,
    Conversation,
    GatewayError,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)


class OpenAICompatConversation(Conversation):
    def __init__(self, client: Any, *, system: str, tools: list[ToolSpec], model: str, max_tokens: int, extra: dict[str, Any]):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.extra = extra
        self.tools = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
            for t in sorted(tools, key=lambda t: t.name)
        ]
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    def send_user(self, text: str) -> AssistantTurn:
        self.messages.append({"role": "user", "content": text})
        return self._call()

    def send_tool_results(self, results: list[ToolResult]) -> AssistantTurn:
        for r in results:
            content = f"ERROR: {r.content}" if r.is_error else r.content
            self.messages.append({"role": "tool", "tool_call_id": r.tool_call_id, "content": content})
        return self._call()

    def _call(self) -> AssistantTurn:
        import openai

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                max_tokens=self.max_tokens,
                **self.extra,
            )
        except openai.RateLimitError as e:
            raise GatewayError(f"rate limited on {self.model}", retryable=True) from e
        except openai.APIStatusError as e:
            raise GatewayError(f"API error {e.status_code} on {self.model}: {e.message}", retryable=e.status_code >= 500) from e
        except openai.APIConnectionError as e:
            raise GatewayError(f"connection error calling {self.model}", retryable=True) from e

        choice = resp.choices[0]
        msg = choice.message
        calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__invalid_json__": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))

        assistant: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        self.messages.append(assistant)

        stop = {"tool_calls": TOOL_USE, "stop": END_TURN, "length": MAX_TOKENS, "content_filter": REFUSAL}.get(
            choice.finish_reason or "", END_TURN
        )
        if calls and stop == END_TURN:
            stop = TOOL_USE
        u = resp.usage
        return AssistantTurn(
            text=msg.content or "",
            tool_calls=calls,
            stop_reason=stop,
            usage=Usage(input_tokens=getattr(u, "prompt_tokens", 0) or 0, output_tokens=getattr(u, "completion_tokens", 0) or 0),
            model=getattr(resp, "model", self.model),
            request_id=getattr(resp, "id", None),
        )


class OpenAICompatBackend(Backend):
    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        effort_style: str | None = None,
        extra: dict[str, Any] | None = None,
        client: Any = None,
    ):
        """``effort_style``: "openrouter" sends ``reasoning: {effort}``, "openai" sends ``reasoning_effort``, None sends nothing."""
        self.effort_style = effort_style
        if client is None:
            try:
                import openai
            except ImportError as e:  # pragma: no cover - optional dependency
                raise GatewayError("install the 'openrouter' extra: pip install -e '.[openrouter]'") from e
            key = os.environ.get(api_key_env)
            if not key:
                raise GatewayError(f"{api_key_env} is not set")
            client = openai.OpenAI(base_url=base_url, api_key=key)
        self.client = client
        self.extra = extra or {}

    def start(self, *, system: str, tools: list[ToolSpec], model: str, effort: str | None = None, max_tokens: int = 16000) -> Conversation:
        extra = dict(self.extra)
        level = effort if effort in ("low", "medium", "high") else ("high" if effort else None)
        if level and self.effort_style == "openrouter":
            # OpenRouter normalizes reasoning effort across providers.
            extra["extra_body"] = {**extra.get("extra_body", {}), "reasoning": {"effort": level}}
        elif level and self.effort_style == "openai":
            extra["reasoning_effort"] = level
        return OpenAICompatConversation(self.client, system=system, tools=tools, model=model, max_tokens=max_tokens, extra=extra)
