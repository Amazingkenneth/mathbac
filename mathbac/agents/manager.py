"""Manager: owns the task DAG, dispatches the ready frontier in parallel, and replans.

Plan modes:
- ``reference``: run the task's reference DAG as written. No LLM planning. Used to
  debug the harness and as the control condition.
- ``llm``: an LLM manager builds and edits the DAG with tools, MACU-style. It never
  sees the reference DAG, which is what its plan is compared against.
"""

from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import ValidationError

from ..dag import DAGError, TaskDAG
from ..gateway import GatewayError, ModelRouter, ToolResult, ToolSpec
from ..ledger import Ledger
from ..schemas import AgentMessage, Claim, NodeResult, NodeSpec, NodeStatus
from .roles import RoleBook, manager_prompt
from .worker import log_turn


@dataclass
class RunSummary:
    status: str
    summary: str = ""
    nodes: dict[str, str] = field(default_factory=dict)


NODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["id", "role", "goal"],
    "properties": {
        "id": {"type": "string"},
        "role": {"type": "string"},
        "goal": {"type": "string"},
        "depends_on": {"type": "array", "items": {"type": "string"}},
        "inputs": {"type": "array", "items": {"type": "string"}},
        "outputs": {"type": "array", "items": {"type": "object", "properties": {"path": {"type": "string"}, "schema": {"type": "string"}}, "required": ["path"]}},
        "acceptance": {"type": "string"},
        "tools": {"type": "array", "items": {"type": "string"}},
        "facts": {"type": "object"},
        "budget": {"type": "object", "properties": {"max_turns": {"type": "integer"}, "max_tokens": {"type": "integer"}, "wall_minutes": {"type": "number"}}},
    },
}


class Manager:
    def __init__(
        self,
        *,
        dag: TaskDAG,
        run_node: Callable[[NodeSpec, int, str], NodeResult],
        ledger: Ledger,
        roles: RoleBook,
        plan_mode: str = "reference",
        router: ModelRouter | None = None,
        task_brief: str = "",
        max_parallel: int = 3,
        max_attempts: int = 2,
        max_manager_turns: int = 80,
        secrets: list[str] | None = None,
    ):
        if plan_mode not in ("reference", "llm"):
            raise ValueError("plan_mode must be 'reference' or 'llm'")
        if plan_mode == "llm" and router is None:
            raise ValueError("llm plan mode needs a model router")
        self.dag = dag
        self.run_node = run_node
        self.ledger = ledger
        self.roles = roles
        self.plan_mode = plan_mode
        self.router = router
        self.task_brief = task_brief
        self.max_parallel = max_parallel
        self.max_attempts = max_attempts
        self.max_manager_turns = max_manager_turns
        self.secrets = secrets or []
        self._attempts: dict[str, int] = {}
        self._running: dict[Future, str] = {}
        self._finished_summary: str | None = None

    # ------------------------------------------------------------------
    def run(self) -> RunSummary:
        self.ledger.append("run_start", plan_mode=self.plan_mode, role_mode=self.roles.mode, dag=self.dag.to_dict())
        with ThreadPoolExecutor(max_workers=self.max_parallel, thread_name_prefix="node") as pool:
            self._pool = pool
            if self.plan_mode == "reference":
                summary = self._run_reference()
            else:
                summary = self._run_llm()
        self.ledger.append("run_end", status=summary.status, summary=summary.summary, nodes=summary.nodes, dag=self.dag.to_dict())
        return summary

    # ---- dispatch -------------------------------------------------------
    def _dispatch_ready(self, feedback: dict[str, str] | None = None) -> None:
        running_ids = set(self._running.values())
        for spec in self.dag.ready():
            if len(self._running) >= self.max_parallel:
                break
            if spec.id in running_ids:
                continue
            attempt = self._attempts.get(spec.id, 0) + 1
            self._attempts[spec.id] = attempt
            self.dag.mark(spec.id, NodeStatus.RUNNING)
            facts = [Claim(key=k, value=v, evidence=f"node:{spec.id}#facts") for k, v in spec.facts.items()]
            self.ledger.append("message", **AgentMessage(sender="manager", recipient=f"{spec.role}@{spec.id}", kind="spec",
                                                         text=spec.goal, claims=facts, node_id=spec.id).model_dump(mode="json"))
            self.ledger.append("node_status", node_id=spec.id, status="running", attempt=attempt, spec=spec)
            fb = (feedback or {}).get(spec.id, "")
            fut = self._pool.submit(self.run_node, spec, attempt, fb)
            self._running[fut] = spec.id

    def _collect(self) -> list[NodeResult]:
        done, _ = wait(list(self._running), return_when=FIRST_COMPLETED)
        results = []
        for fut in done:
            nid = self._running.pop(fut)
            try:
                res = fut.result()
            except Exception as e:  # a crashed worker is a failed node, never a crashed run
                res = NodeResult(node_id=nid, status=NodeStatus.FAILED, error=f"worker crashed: {type(e).__name__}: {e}",
                                 attempts=self._attempts.get(nid, 1))
            self.dag.mark(nid, res.status, res)
            self.ledger.append("node_status", node_id=nid, status=res.status.value, attempt=res.attempts,
                               error=res.error, summary=res.summary, outputs=res.outputs)
            results.append(res)
        return results

    def _status_map(self) -> dict[str, str]:
        return {i: self.dag.status(i).value for i in self.dag.ids()}

    # ---- reference mode ----------------------------------------------
    def _run_reference(self) -> RunSummary:
        feedback: dict[str, str] = {}
        while True:
            self._dispatch_ready(feedback)
            if not self._running:
                break
            for res in self._collect():
                if res.status == NodeStatus.FAILED and self._attempts.get(res.node_id, 1) < self.max_attempts:
                    self.dag.rewrite(self.dag.spec(res.node_id))  # back to pending for a retry
                    feedback[res.node_id] = f"Attempt {res.attempts} failed: {res.error}"
        statuses = self._status_map()
        ok = all(s == "done" for s in statuses.values())
        return RunSummary(status="complete" if ok else "incomplete", nodes=statuses,
                          summary="" if ok else f"blocked: {', '.join(self.dag.blocked())}")

    # ---- llm mode ------------------------------------------------------
    def _manager_tools(self) -> list[ToolSpec]:
        tools = [
            ToolSpec("add_node", "Add a node to the task graph. Its dependencies must already exist.",
                     {"type": "object", "properties": {"node": NODE_SCHEMA}, "required": ["node"]}),
            ToolSpec("rewrite_node", "Replace a pending or failed node's spec. It goes back to pending.",
                     {"type": "object", "properties": {"node": NODE_SCHEMA}, "required": ["node"]}),
            ToolSpec("cancel_node", "Cancel a pending or failed node and everything downstream of it.",
                     {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}),
            ToolSpec("view_dag", "Show every node with its status.", {"type": "object", "properties": {}}),
            ToolSpec("finish_run", "End the run once the task's outputs exist, or when nothing more can be done.",
                     {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}),
        ]
        if self.roles.mode == "dynamic":
            tools.append(ToolSpec("define_role", "Write a role prompt that worker agents with this role will follow.",
                                  {"type": "object", "properties": {"name": {"type": "string"}, "prompt": {"type": "string"}},
                                   "required": ["name", "prompt"]}))
        return tools

    def _manager_call(self, name: str, inp: dict[str, Any]) -> tuple[str, bool]:
        try:
            if name in ("add_node", "rewrite_node"):
                spec = NodeSpec.model_validate(inp["node"])
                if self.roles.mode == "predefined" and spec.role not in self.roles.names():
                    return f"unknown role {spec.role!r}; available roles: {', '.join(self.roles.names())}", True
                (self.dag.add if name == "add_node" else self.dag.rewrite)(spec)
                self.ledger.append("dag_edit", op=name, node=spec)
                if name == "rewrite_node":
                    self._attempts[spec.id] = 0
                return f"{name}: {spec.id} ok", False
            if name == "cancel_node":
                ids = self.dag.cancel(inp["id"])
                self.ledger.append("dag_edit", op="cancel_node", ids=ids)
                return f"cancelled {ids}", False
            if name == "view_dag":
                return self.dag.render() or "(empty)", False
            if name == "define_role":
                self.roles.define(inp["name"], inp["prompt"])
                self.ledger.append("dag_edit", op="define_role", name=inp["name"], prompt=inp["prompt"])
                return f"role {inp['name']} defined", False
            if name == "finish_run":
                self._finished_summary = str(inp.get("summary", ""))
                return "run will end", False
            return f"unknown tool {name!r}", True
        except (DAGError, ValidationError, KeyError, PermissionError) as e:
            return f"{type(e).__name__}: {e}", True

    def _drain(self, conv, turn) -> Any:
        """Execute manager tool calls until it yields control (a turn with no tool calls)."""
        while True:
            log_turn(self.ledger, "manager", "manager", None, turn, self.secrets)
            self._manager_turns += 1
            if not turn.tool_calls or self._finished_summary is not None:
                return turn
            results = []
            for c in turn.tool_calls:
                out, err = self._manager_call(c.name, c.input)
                self.ledger.append("tool_call", agent="manager", node_id=None, tool=c.name,
                                   input=json.dumps(c.input, default=str)[:2000], is_error=err, output=out[:2000], seconds=0.0)
                results.append(ToolResult(c.id, out, err))
            if self._finished_summary is not None or self._manager_turns >= self.max_manager_turns:
                return turn
            turn = conv.send_tool_results(results)

    def _run_llm(self) -> RunSummary:
        self._manager_turns = 0
        roles_txt = ", ".join(self.roles.names()) if self.roles.mode == "predefined" else \
            "none yet: define each role you use with define_role before adding nodes that use it"
        conv = self.router.start("manager", system=manager_prompt(), tools=self._manager_tools())
        first = f"{self.task_brief}\n\nWorker roles: {roles_txt}.\nAt most {self.max_parallel} nodes run at the same time."
        try:
            turn = self._drain(conv, conv.send_user(first))
            idle_nudges = 0
            while self._finished_summary is None and self._manager_turns < self.max_manager_turns:
                self._dispatch_ready()
                if not self._running:
                    if idle_nudges >= 2:
                        break
                    idle_nudges += 1
                    blocked = self.dag.blocked()
                    msg = ("No node is running and none is ready. "
                           + (f"Blocked by failed upstream nodes: {', '.join(blocked)}. " if blocked else "")
                           + "Add or rewrite nodes, or call finish_run.\n\n" + (self.dag.render() or "(graph is empty)"))
                    turn = self._drain(conv, conv.send_user(msg))
                    continue
                idle_nudges = 0
                events = []
                for res in self._collect():
                    claims = "\n".join(f"  - {c.key} = {c.value!r} (evidence: {c.evidence})" for c in res.claims)
                    events.append(f"Node {res.node_id} {res.status.value} (attempt {res.attempts}).\n"
                                  f"Summary: {res.summary or res.error}\nOutputs: {res.outputs}\nClaims:\n{claims or '  (none)'}")
                turn = self._drain(conv, conv.send_user("\n\n".join(events) + "\n\nCurrent graph:\n" + self.dag.render()))
        except GatewayError as e:
            # Let running nodes finish so their outputs are not lost, then report.
            while self._running:
                self._collect()
            return RunSummary(status="manager_error", summary=str(e), nodes=self._status_map())
        while self._running:
            self._collect()
        statuses = self._status_map()
        ok = bool(statuses) and all(s in ("done", "cancelled") for s in statuses.values())
        return RunSummary(status="complete" if ok else "incomplete", summary=self._finished_summary or "", nodes=statuses)
