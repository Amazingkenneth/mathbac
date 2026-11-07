"""Registry of scientific tools described by tool cards (``toolcards/*.yaml``)."""

from __future__ import annotations

import shlex
from pathlib import Path

import yaml

from ..schemas import ToolCard


class ToolRegistryError(ValueError):
    pass


class ToolRegistry:
    def __init__(self, cards: dict[str, ToolCard]):
        self.cards = cards

    @classmethod
    def load(cls, directory: Path) -> "ToolRegistry":
        cards: dict[str, ToolCard] = {}
        for p in sorted(Path(directory).glob("*.yaml")):
            card = ToolCard.model_validate(yaml.safe_load(p.read_text()))
            if card.name in cards:
                raise ToolRegistryError(f"duplicate tool card name {card.name!r} in {p}")
            cards[card.name] = card
        return cls(cards)

    def get(self, name: str) -> ToolCard:
        if name not in self.cards:
            raise ToolRegistryError(f"no tool named {name!r}. Known tools: {', '.join(sorted(self.cards))}")
        return self.cards[name]

    def summary(self, allowed: list[str] | None = None) -> str:
        rows = []
        for c in sorted(self.cards.values(), key=lambda c: (c.tier, c.name)):
            if allowed is not None and c.name not in allowed:
                continue
            state = "installed" if c.installed else "card only"
            rows.append(f"- {c.name} (tier {c.tier}, {state}): {c.purpose}")
        return "\n".join(rows) or "(no tools available to this node)"

    def render(self, name: str) -> str:
        c = self.get(name)
        out = [f"# {c.title}  [{c.name}, tier {c.tier}]", "", c.purpose, ""]
        if c.source:
            out.append(f"Source: {c.source}" + (f"  (doi:{c.doi})" if c.doi else ""))
        out.append(f"Installed in this environment: {'yes' if c.installed else 'no'}")
        if c.invoke:
            out.append(f"Invocation template: {c.invoke}")
        if c.args:
            out.append("Arguments:")
            for k, a in c.args.items():
                req = "required" if a.required else f"optional, default {a.default!r}"
                out.append(f"  - {k} ({a.type}, {req}): {a.description}")
        for label, val in (("Inputs", c.inputs), ("Outputs", c.outputs), ("Use for protein-NP work", c.pni_translatability), ("Cost", c.cost)):
            if val:
                out.append(f"{label}: {val}")
        if c.preconditions:
            out.append("Preconditions:")
            out += [f"  - {p}" for p in c.preconditions]
        if c.pitfalls:
            out.append("Pitfalls (read these):")
            out += [f"  - {p}" for p in c.pitfalls]
        if c.smoke_test:
            out.append(f"Smoke test: {c.smoke_test}")
        return "\n".join(out)

    def command(self, name: str, args: dict) -> str:
        c = self.get(name)
        if not c.installed or not c.invoke:
            raise ToolRegistryError(
                f"{name} is described by a card but not installed in this environment. "
                "Ask the manager for an alternative, or work without it."
            )
        values = {}
        for k, a in c.args.items():
            if k in args:
                values[k] = args[k]
            elif a.required:
                raise ToolRegistryError(f"{name}: missing required argument {k!r}")
            else:
                values[k] = a.default
        unknown = set(args) - set(c.args)
        if unknown:
            raise ToolRegistryError(f"{name}: unknown arguments {sorted(unknown)}")
        quoted = {}
        for k, v in values.items():
            if c.args[k].type == "flags":
                # A space-separated list of --switches, each quoted on its own; anything else is rejected.
                parts = str(v or "").split()
                if any(not p.startswith("--") for p in parts):
                    raise ToolRegistryError(f"{name}: {k} accepts only --switches, got {v!r}")
                quoted[k] = " ".join(shlex.quote(p) for p in parts)
            else:
                quoted[k] = shlex.quote(str(v))
        try:
            return c.invoke.format(**quoted)
        except KeyError as e:
            raise ToolRegistryError(f"{name}: invoke template references undefined argument {e}") from e
