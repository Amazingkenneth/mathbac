"""Model gateway: maps each agent role to a provider, model and effort level."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from .base import AssistantTurn, Backend, Conversation, GatewayError, ToolCall, ToolResult, ToolSpec, Usage

# $ per million tokens (input, output). Claude prices from Anthropic's published table; others unknown here.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-1": (10.0, 50.0),
}


def estimate_cost(model: str, usage: Usage) -> float | None:
    p = PRICES.get(model)
    if p is None:
        return None
    # Cache reads bill at ~0.1x input, cache writes at ~1.25x input.
    inp = usage.input_tokens + 0.1 * usage.cache_read_tokens + 1.25 * usage.cache_write_tokens
    return (inp * p[0] + usage.output_tokens * p[1]) / 1e6


@dataclass
class RoleModel:
    provider: str
    model: str
    effort: str | None
    max_tokens: int


class ModelRouter:
    """Reads ``configs/models.yaml``: providers, defaults, and per-role overrides."""

    def __init__(self, config: dict[str, Any], backends: dict[str, Backend] | None = None):
        self.config = config
        self._backends: dict[str, Backend] = dict(backends or {})

    def for_role(self, role: str) -> RoleModel:
        d = {**self.config.get("defaults", {}), **(self.config.get("roles", {}).get(role) or {})}
        return RoleModel(
            provider=d.get("provider", "anthropic"),
            model=d.get("model", "claude-opus-5-5"),
            effort=d.get("effort"),
            max_tokens=int(d.get("max_tokens", 16000)),
        )

    def backend(self, provider: str) -> Backend:
        if provider not in self._backends:
            pcfg = self.config.get("providers", {}).get(provider)
            if pcfg is None:
                raise GatewayError(f"provider {provider!r} is not configured in models.yaml")
            kind = pcfg.get("type", provider)
            if kind == "anthropic":
                from .anthropic_backend import AnthropicBackend

                self._backends[provider] = AnthropicBackend(
                    api_key_env=pcfg.get("api_key_env", "ANTHROPIC_API_KEY"),
                    fallbacks=pcfg.get("fallbacks", "default") == "default",
                )
            elif kind == "openai_compat":
                from .openai_compat_backend import OpenAICompatBackend

                self._backends[provider] = OpenAICompatBackend(
                    base_url=pcfg["base_url"],
                    api_key_env=pcfg["api_key_env"],
                    effort_style=pcfg.get("effort_style"),
                )
            else:
                raise GatewayError(f"unknown provider type {kind!r}")
        return self._backends[provider]

    def start(self, role: str, *, system: str, tools: list[ToolSpec]) -> Conversation:
        rm = self.for_role(role)
        return self.backend(rm.provider).start(system=system, tools=tools, model=rm.model, effort=rm.effort, max_tokens=rm.max_tokens)

    def secret_values(self) -> list[str]:
        envs = {p.get("api_key_env") for p in self.config.get("providers", {}).values()}
        envs |= {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENROUTER_API_KEY", "OPENAI_API_KEY"}
        return [v for e in envs if e and (v := os.environ.get(e))]


_KEY_PATTERNS = [re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"), re.compile(r"sk-or-[A-Za-z0-9_\-]{8,}"), re.compile(r"sk-[A-Za-z0-9]{20,}")]


def redact(text: str, secrets: list[str] = ()) -> str:
    """Remove API keys from anything that is logged or shown to a model."""
    for s in secrets:
        if s and len(s) >= 8:
            text = text.replace(s, "[REDACTED]")
    for pat in _KEY_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


__all__ = [
    "AssistantTurn",
    "Backend",
    "Conversation",
    "GatewayError",
    "ModelRouter",
    "RoleModel",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "estimate_cost",
    "redact",
]
