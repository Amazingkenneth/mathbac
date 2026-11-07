"""Claude backend built on the official Anthropic Python SDK."""

from __future__ import annotations

import os
from typing import Any

import anthropic

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

# Models that accept the server-side refusal fallback in its "default" form on the Claude API.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicConversation(Conversation):
    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        system: str,
        tools: list[ToolSpec],
        model: str,
        effort: str | None,
        max_tokens: int,
        fallbacks: bool,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        # Stable prefix first (tools, then system) so prompt caching hits on every turn.
        self.system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        self.tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in sorted(tools, key=lambda t: t.name)
        ]
        self.output_config: dict[str, Any] = {"effort": effort} if effort else {}
        self.use_fallbacks = fallbacks and model in FALLBACK_MODELS
        self.messages: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def send_user(self, text: str) -> AssistantTurn:
        self.messages.append({"role": "user", "content": text})
        return self._call()

    def send_tool_results(self, results: list[ToolResult]) -> AssistantTurn:
        # All results for one assistant turn go back in a single user message.
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.tool_call_id,
                        "content": r.content,
                        **({"is_error": True} if r.is_error else {}),
                    }
                    for r in results
                ],
            }
        )
        return self._call()

    # ------------------------------------------------------------------
    def _request(self) -> Any:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
            thinking={"type": "adaptive"},
            cache_control={"type": "ephemeral"},
        )
        if self.output_config:
            kwargs["output_config"] = self.output_config
        if self.use_fallbacks:
            return self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        return self.client.messages.create(**kwargs)

    def _call(self) -> AssistantTurn:
        usage = Usage()
        while True:
            try:
                response = self._request()
            except anthropic.BadRequestError as e:
                raise GatewayError(f"bad request to {self.model}: {e.message}") from e
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
                raise GatewayError(f"credentials rejected for {self.model}: {e.message}") from e
            except anthropic.NotFoundError as e:
                raise GatewayError(f"model or endpoint not found: {self.model}") from e
            except anthropic.RateLimitError as e:
                raise GatewayError(f"rate limited on {self.model} after SDK retries", retryable=True) from e
            except anthropic.APIStatusError as e:
                raise GatewayError(f"API error {e.status_code} on {self.model}: {e.message}", retryable=e.status_code >= 500) from e
            except anthropic.APIConnectionError as e:
                raise GatewayError(f"connection error calling {self.model}", retryable=True) from e

            u = response.usage
            usage += Usage(
                input_tokens=u.input_tokens or 0,
                output_tokens=u.output_tokens or 0,
                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            )
            # Append the full content (thinking, fallback and tool_use blocks included), never just the text.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue
            break

        text = "\n".join(b.text for b in response.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, input=dict(b.input)) for b in response.content if b.type == "tool_use"]
        stop = {
            "end_turn": END_TURN,
            "tool_use": TOOL_USE,
            "max_tokens": MAX_TOKENS,
            "refusal": REFUSAL,
            "stop_sequence": END_TURN,
        }.get(response.stop_reason, response.stop_reason)
        notes: dict[str, Any] = {}
        if response.stop_reason == "refusal" and getattr(response, "stop_details", None):
            notes["refusal_category"] = response.stop_details.category
            notes["refusal_explanation"] = response.stop_details.explanation
        served_by = getattr(response, "model", self.model)
        return AssistantTurn(
            text=text,
            tool_calls=calls,
            stop_reason=stop,
            usage=usage,
            model=served_by,
            request_id=getattr(response, "_request_id", None),
            notes=notes,
        )


class AnthropicBackend(Backend):
    def __init__(self, api_key_env: str = "ANTHROPIC_API_KEY", fallbacks: bool = True, client: anthropic.Anthropic | None = None):
        if client is None:
            key = os.environ.get(api_key_env)
            # With no key set, the SDK still resolves ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile.
            client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        self.client = client
        self.fallbacks = fallbacks

    def start(self, *, system: str, tools: list[ToolSpec], model: str, effort: str | None = None, max_tokens: int = 16000) -> Conversation:
        return AnthropicConversation(
            self.client,
            system=system,
            tools=tools,
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            fallbacks=self.fallbacks,
        )
