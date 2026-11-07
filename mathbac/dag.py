"""Task dependency graph with MACU-style operations.

The manager adds, cancels and rewrites nodes while the run is in progress. Workers
are dispatched to the *ready frontier*: pending nodes whose dependencies are all done.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterable

import yaml

from .schemas import NodeResult, NodeSpec, NodeStatus


class DAGError(ValueError):
    pass


class TaskDAG:
    def __init__(self, nodes: Iterable[NodeSpec] = ()):
        self._specs: dict[str, NodeSpec] = {}
        self._status: dict[str, NodeStatus] = {}
        self._results: dict[str, NodeResult] = {}
        self._lock = threading.RLock()
        for n in nodes:
            self.add(n)

    # ---- construction -------------------------------------------------
    @classmethod
    def from_yaml(cls, path: Path) -> "TaskDAG":
        data = yaml.safe_load(Path(path).read_text())
        return cls(NodeSpec.model_validate(n) for n in data["nodes"])

    def subset(self, keep: Iterable[str]) -> "TaskDAG":
        """A new DAG with only ``keep``. Dependencies on dropped nodes are removed, so those
        nodes' outputs must already be in the workspace (e.g. injected reference artifacts)."""
        keep = set(keep)
        unknown = keep - set(self._specs)
        if unknown:
            raise DAGError(f"unknown nodes {sorted(unknown)}")
        out = TaskDAG()
        for i in self.order():
            if i in keep:
                s = self._specs[i]
                out.add(s.model_copy(update={"depends_on": [d for d in s.depends_on if d in keep]}))
        return out

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "nodes": [
                    {
                        **self._specs[i].model_dump(mode="json", by_alias=True),
                        "status": self._status[i].value,
                    }
                    for i in self.order()
                ]
            }

    # ---- edits (manager operations) -----------------------------------
    def add(self, spec: NodeSpec) -> None:
        with self._lock:
            if spec.id in self._specs:
                raise DAGError(f"node {spec.id!r} already exists")
            missing = [d for d in spec.depends_on if d not in self._specs]
            if missing:
                raise DAGError(f"node {spec.id!r} depends on unknown nodes {missing}")
            self._specs[spec.id] = spec
            self._status[spec.id] = NodeStatus.PENDING

    def rewrite(self, spec: NodeSpec) -> None:
        """Replace a pending or failed node's spec (e.g. after a failure) and reset it to pending."""
        with self._lock:
            if spec.id not in self._specs:
                raise DAGError(f"unknown node {spec.id!r}")
            if self._status[spec.id] in (NodeStatus.RUNNING, NodeStatus.DONE):
                raise DAGError(f"cannot rewrite node {spec.id!r} while it is {self._status[spec.id].value}")
            missing = [d for d in spec.depends_on if d not in self._specs]
            if missing:
                raise DAGError(f"node {spec.id!r} depends on unknown nodes {missing}")
            old = self._specs[spec.id]
            self._specs[spec.id] = spec
            if self._has_cycle():
                self._specs[spec.id] = old
                raise DAGError(f"rewriting {spec.id!r} would create a cycle")
            self._status[spec.id] = NodeStatus.PENDING

    def cancel(self, node_id: str) -> list[str]:
        """Cancel a node and everything downstream of it. Returns the cancelled ids."""
        with self._lock:
            if node_id not in self._specs:
                raise DAGError(f"unknown node {node_id!r}")
            cancelled = []
            for nid in [node_id, *self.descendants(node_id)]:
                if self._status[nid] in (NodeStatus.PENDING, NodeStatus.FAILED):
                    self._status[nid] = NodeStatus.CANCELLED
                    cancelled.append(nid)
            return cancelled

    # ---- scheduling ----------------------------------------------------
    def ready(self) -> list[NodeSpec]:
        with self._lock:
            return [
                self._specs[i]
                for i in self.order()
                if self._status[i] == NodeStatus.PENDING
                and all(self._status[d] == NodeStatus.DONE for d in self._specs[i].depends_on)
            ]

    def blocked(self) -> list[str]:
        """Pending nodes that can never run because an upstream node failed or was cancelled."""
        with self._lock:
            bad = {i for i, s in self._status.items() if s in (NodeStatus.FAILED, NodeStatus.CANCELLED)}
            return [
                i
                for i, s in self._status.items()
                if s == NodeStatus.PENDING and any(a in bad for a in self.ancestors(i))
            ]

    def mark(self, node_id: str, status: NodeStatus, result: NodeResult | None = None) -> None:
        with self._lock:
            self._status[node_id] = status
            if result is not None:
                self._results[node_id] = result

    def is_finished(self) -> bool:
        with self._lock:
            if any(s == NodeStatus.RUNNING for s in self._status.values()):
                return False
            return not self.ready()

    # ---- queries -------------------------------------------------------
    def spec(self, node_id: str) -> NodeSpec:
        return self._specs[node_id]

    def status(self, node_id: str) -> NodeStatus:
        return self._status[node_id]

    def result(self, node_id: str) -> NodeResult | None:
        return self._results.get(node_id)

    def ids(self) -> list[str]:
        return list(self._specs)

    def ancestors(self, node_id: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self._specs[node_id].depends_on)
        while stack:
            n = stack.pop()
            if n not in seen:
                seen.add(n)
                stack.extend(self._specs[n].depends_on)
        return seen

    def descendants(self, node_id: str) -> list[str]:
        out = [i for i in self._specs if node_id in self.ancestors(i)]
        return [i for i in self.order() if i in out]

    def order(self) -> list[str]:
        """Topological order, stable with respect to insertion order."""
        indeg = {i: 0 for i in self._specs}
        for i, s in self._specs.items():
            for _ in s.depends_on:
                indeg[i] += 1
        queue = [i for i in self._specs if indeg[i] == 0]
        out: list[str] = []
        while queue:
            n = queue.pop(0)
            out.append(n)
            for i, s in self._specs.items():
                if n in s.depends_on:
                    indeg[i] -= 1
                    if indeg[i] == 0:
                        queue.append(i)
        if len(out) != len(self._specs):
            raise DAGError("task graph has a cycle")
        return out

    def _has_cycle(self) -> bool:
        try:
            self.order()
            return False
        except DAGError:
            return True

    def render(self) -> str:
        lines = []
        for i in self.order():
            s = self._specs[i]
            deps = ",".join(s.depends_on) or "-"
            lines.append(f"[{self._status[i].value:9}] {i:6} role={s.role:20} deps={deps:12} {s.goal[:70]}")
        return "\n".join(lines)
