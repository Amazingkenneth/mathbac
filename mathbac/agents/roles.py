"""Role prompts.

Mode "predefined" uses the prompts shipped in ``agents/prompts``. Mode "dynamic"
starts with no worker roles: the manager writes them with ``define_role``. Both
modes run on the same harness so experiment E2 compares only the role source.
"""

from __future__ import annotations

import threading
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"
PREDEFINED = ["spec_reader", "env_engineer", "data_curator", "descriptor_engineer", "ml_engineer", "reporter"]


def common_prompt() -> str:
    return (PROMPT_DIR / "_common.md").read_text()


def manager_prompt() -> str:
    return (PROMPT_DIR / "manager.md").read_text()


def verifier_prompt() -> str:
    return (PROMPT_DIR / "verifier.md").read_text()


class RoleBook:
    def __init__(self, mode: str = "predefined"):
        if mode not in ("predefined", "dynamic"):
            raise ValueError("role mode must be 'predefined' or 'dynamic'")
        self.mode = mode
        self._roles: dict[str, str] = {}
        self._lock = threading.Lock()
        if mode == "predefined":
            for r in PREDEFINED:
                self._roles[r] = (PROMPT_DIR / f"{r}.md").read_text()

    def define(self, name: str, prompt: str) -> None:
        if self.mode != "dynamic":
            raise PermissionError("roles are fixed in predefined mode")
        with self._lock:
            self._roles[name] = prompt

    def get(self, name: str) -> str:
        with self._lock:
            if name not in self._roles:
                raise KeyError(f"unknown role {name!r}; defined roles: {', '.join(self._roles) or 'none'}")
            return self._roles[name]

    def names(self) -> list[str]:
        with self._lock:
            return list(self._roles)
