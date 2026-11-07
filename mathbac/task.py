"""Loads a reproduction task (``tasks/<id>/task.yaml``) and its in-loop acceptance checks."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from .dag import TaskDAG
from .schemas import NodeSpec

AcceptanceFn = Callable[[Path, NodeSpec], list[str]]  # returns a list of problems; empty means pass

ACCESS_LEVELS = ("L1", "L2", "L3")


@dataclass
class Task:
    root: Path
    config: dict

    @classmethod
    def load(cls, root: Path) -> "Task":
        root = Path(root).resolve()
        return cls(root=root, config=yaml.safe_load((root / "task.yaml").read_text()))

    @property
    def id(self) -> str:
        return self.config["id"]

    @property
    def gold_dir(self) -> Path:
        return Path(__file__).parent / "eval" / "gold" / self.config.get("gold", self.id)

    def brief(self) -> str:
        return (self.root / self.config.get("brief", "brief.md")).read_text()

    def contracts(self) -> str:
        return (self.root / self.config.get("contracts", "contracts/artifacts.md")).read_text()

    def reference_dag(self) -> TaskDAG:
        return TaskDAG.from_yaml(self.root / self.config.get("reference_dag", "reference_dag.yaml"))

    def materials_for(self, level: str, materials_root: Path) -> dict[str, Path]:
        """Materials folders visible at an access level, e.g. L2 -> paper, si, code."""
        if level not in ACCESS_LEVELS:
            raise ValueError(f"access level must be one of {ACCESS_LEVELS}")
        names = self.config["access_levels"][level]
        out = {}
        for n in names:
            p = Path(materials_root) / n
            if p.exists():
                out[n] = p
        return out

    def acceptance(self, name: str) -> AcceptanceFn:
        path = self.root / "acceptance" / f"{name}.py"
        if not path.exists():
            raise FileNotFoundError(f"acceptance check {name!r} not found at {path}")
        spec = importlib.util.spec_from_file_location(f"acceptance_{self.id}_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod.check
