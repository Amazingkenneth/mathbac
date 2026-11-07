"""Worker agent: runs one DAG node as a tool-calling loop in the sandbox."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..gateway import AssistantTurn, GatewayError, ModelRouter, ToolResult, ToolSpec, Usage, estimate_cost, redact
from ..gateway.base import MAX_TOKENS, REFUSAL
from ..ledger import Ledger
from ..schemas import AgentMessage, Claim, NodeResult, NodeSpec, NodeStatus
from ..tools import AgentTool, ToolContext, run_handler, worker_tools
from .roles import RoleBook, common_prompt, verifier_prompt
from .verifier import ContractVerifier


@dataclass
class LoopOutcome:
    control: str | None  # name of the control tool that ended the loop, or None on failure
    control_input: dict[str, Any] | None
    usage: Usage
    turns: int
    error: str | None = None


ControlFn = Callable[[dict[str, Any]], str | None]  # returns None to accept and end, or feedback text to continue


def log_turn(ledger: Ledger, who: str, role: str, node_id: str | None, turn: AssistantTurn, secrets: list[str]) -> None:
    ledger.append(
        "model_turn",
        agent=who,
        role=role,
        node_id=node_id,
        model=turn.model,
        stop_reason=turn.stop_reason,
        text=redact(turn.text, secrets)[:4000],
        tool_calls=[{"name": c.name, "id": c.id} for c in turn.tool_calls],
        usage=turn.usage.__dict__,
        cost_usd=estimate_cost(turn.model, turn.usage),
        request_id=turn.request_id,
        notes=turn.notes,
    )


def run_loop(
    *,
    conv,
    first_message: str,
    tools: list[AgentTool],
    ctx: ToolContext,
    controls: dict[str, ControlFn],
    ledger: Ledger,
    who: str,
    role: str,
    max_turns: int,
    max_tokens: int,
    wall_seconds: float,
    max_nudges: int = 2,
) -> LoopOutcome:
    """Drive a conversation until a control tool accepts, or a limit is hit."""
    by_name = {t.spec.name: t for t in tools}
    usage = Usage()
    t0 = time.time()
    nudges = 0
    turns = 0
    try:
        turn = conv.send_user(first_message)
        while True:
            turns += 1
            usage += turn.usage
            log_turn(ledger, who, role, ctx.node.id, turn, ctx.secrets)

            if turn.stop_reason == REFUSAL:
                return LoopOutcome(None, None, usage, turns, f"model declined the request: {turn.notes}")

            if not turn.tool_calls:
                if nudges >= max_nudges:
                    return LoopOutcome(None, None, usage, turns, "agent stopped without calling a completion tool")
                nudges += 1
                if turn.stop_reason == MAX_TOKENS:
                    turn = conv.send_user("Your last reply hit the output limit. Continue in smaller steps.")
                else:
                    names = " or ".join(controls)
                    turn = conv.send_user(f"You ended your turn without calling {names}. Keep working, then call {names}.")
                continue

            results: list[ToolResult] = []
            ended: tuple[str, dict] | None = None
            for call in turn.tool_calls:
                if call.name in controls:
                    feedback = controls[call.name](call.input)
                    ledger.append("tool_call", agent=who, node_id=ctx.node.id, tool=call.name,
                                  input=_clip(call.input, ctx.secrets), is_error=feedback is not None,
                                  output=(feedback or "accepted")[:2000], seconds=0.0)
                    if feedback is None:
                        ended = (call.name, call.input)
                        results.append(ToolResult(call.id, "accepted"))
                    else:
                        results.append(ToolResult(call.id, feedback, is_error=True))
                    continue
                tool = by_name.get(call.name)
                if tool is None or tool.handler is None:
                    out, err, secs = f"unknown tool {call.name!r}", True, 0.0
                else:
                    out, err, secs = run_handler(tool, ctx, call.input)
                ledger.append("tool_call", agent=who, node_id=ctx.node.id, tool=call.name,
                              input=_clip(call.input, ctx.secrets), is_error=err, output=out[:2000], seconds=secs)
                results.append(ToolResult(call.id, out, is_error=err))

            if ended:
                return LoopOutcome(ended[0], ended[1], usage, turns)
            if turns >= max_turns:
                return LoopOutcome(None, None, usage, turns, f"turn budget of {max_turns} exhausted")
            if usage.total >= max_tokens:
                return LoopOutcome(None, None, usage, turns, f"token budget of {max_tokens} exhausted")
            if time.time() - t0 > wall_seconds:
                return LoopOutcome(None, None, usage, turns, f"wall-clock budget of {wall_seconds/60:.0f} min exhausted")
            turn = conv.send_tool_results(results)
    except GatewayError as e:
        return LoopOutcome(None, None, usage, turns, f"model gateway error: {e}" + (" (retryable)" if e.retryable else ""))


def _clip(obj: Any, secrets: list[str], limit: int = 2000) -> Any:
    s = redact(json.dumps(obj, default=str), secrets)
    return s if len(s) <= limit else s[:limit] + "...[clipped]"


def node_brief(node: NodeSpec, access_level: str, materials: list[str], registry_summary: str, feedback: str = "") -> str:
    outs = "\n".join(f"- {o.path}" + (f"  (schema: {o.schema_})" if o.schema_ else "") for o in node.outputs) or "- (none declared)"
    facts = json.dumps(node.facts, indent=1) if node.facts else "{}"
    parts = [
        f"Node {node.id} ({node.role})",
        f"Goal: {node.goal}",
        f"Depends on: {', '.join(node.depends_on) or 'nothing'}",
        "Inputs:\n" + ("\n".join(f"- {i}" for i in node.inputs) or "- (none)"),
        "Outputs you must write:\n" + outs,
        f"Method facts from the manager:\n{facts}",
        f"Acceptance check: {node.acceptance or 'output files must exist'}",
        f"Access level {access_level}; materials available: {', '.join(materials) or 'none'}",
        "Scientific tools for this node:\n" + registry_summary,
    ]
    if feedback:
        parts.append(f"Notes from the previous attempt at this node:\n{feedback}")
    return "\n\n".join(parts)


class WorkerAgent:
    def __init__(
        self,
        *,
        router: ModelRouter,
        roles: RoleBook,
        contracts: str,
        verifier: ContractVerifier,
        ledger: Ledger,
        ctx_factory: Callable[[NodeSpec], ToolContext],
        access_level: str,
        llm_review: bool = False,
        max_rejections: int = 3,
    ):
        self.router = router
        self.roles = roles
        self.contracts = contracts
        self.verifier = verifier
        self.ledger = ledger
        self.ctx_factory = ctx_factory
        self.access_level = access_level
        self.llm_review = llm_review
        self.max_rejections = max_rejections

    def run(self, node: NodeSpec, attempt: int = 1, feedback: str = "") -> NodeResult:
        ctx = self.ctx_factory(node)
        who = f"{node.role}@{node.id}"
        try:
            role_prompt = self.roles.get(node.role)
        except KeyError as e:
            return NodeResult(node_id=node.id, status=NodeStatus.FAILED, error=str(e), attempts=attempt)

        system = "\n\n".join([common_prompt(), role_prompt, "# Artifact contracts\n\n" + self.contracts])
        tools = worker_tools()
        conv = self.router.start(node.role, system=system, tools=[t.spec for t in tools])
        brief = node_brief(node, self.access_level, sorted(ctx.materials), ctx.registry.summary(node.tools or None), feedback)

        rejections = 0
        accepted: dict[str, Any] = {}

        def on_finish(inp: dict[str, Any]) -> str | None:
            nonlocal rejections
            claims = _parse_claims(inp.get("claims", []))
            self.ledger.append("message", **AgentMessage(sender=who, recipient="manager", kind="result",
                                                         text=redact(str(inp.get("summary", "")), ctx.secrets),
                                                         claims=claims, node_id=node.id).model_dump(mode="json"))
            for p in inp.get("outputs", []):
                if isinstance(p, str):
                    self.ledger.record_artifact(ctx.workspace, p, node.id)
            problems = self.verifier.check(node)
            if not problems and self.llm_review:
                problems = self._review(node, ctx, inp)
            if problems:
                rejections += 1
                text = "Verifier rejected the outputs:\n" + "\n".join(f"- {p}" for p in problems)
                self.ledger.append("message", **AgentMessage(sender="verifier", recipient=who, kind="rejection",
                                                             text=text, node_id=node.id).model_dump(mode="json"))
                if rejections > self.max_rejections:
                    accepted["give_up"] = text
                    return None
                return text + "\nFix these and call finish again."
            accepted.update(inp)
            accepted["claims_parsed"] = claims
            return None

        b = node.budget
        out = run_loop(conv=conv, first_message=brief, tools=tools, ctx=ctx, controls={"finish": on_finish},
                       ledger=self.ledger, who=who, role=node.role, max_turns=b.max_turns, max_tokens=b.max_tokens,
                       wall_seconds=b.wall_minutes * 60)

        if out.control == "finish" and "give_up" not in accepted:
            return NodeResult(node_id=node.id, status=NodeStatus.DONE, summary=str(accepted.get("summary", "")),
                              outputs=[p for p in accepted.get("outputs", []) if isinstance(p, str)],
                              claims=accepted.get("claims_parsed", []), attempts=attempt)
        err = accepted.get("give_up") or out.error or "unknown failure"
        return NodeResult(node_id=node.id, status=NodeStatus.FAILED, error=err, attempts=attempt)

    # ------------------------------------------------------------------
    def _review(self, node: NodeSpec, ctx: ToolContext, finish_input: dict[str, Any]) -> list[str]:
        read_only = [t for t in worker_tools() if t.spec.name in ("read_file", "list_dir")]
        verdict_spec = ToolSpec("verdict", "Accept or reject the node's outputs.", {
            "type": "object", "additionalProperties": False, "required": ["accept", "reasons"],
            "properties": {"accept": {"type": "boolean"}, "reasons": {"type": "array", "items": {"type": "string"}}},
        })
        tools = read_only + [AgentTool(verdict_spec, None)]
        conv = self.router.start("verifier", system=common_prompt() + "\n\n" + verifier_prompt(), tools=[t.spec for t in tools])
        msg = node_brief(node, self.access_level, sorted(ctx.materials), "") + \
            "\n\nThe worker's finish call:\n" + json.dumps(finish_input, indent=1, default=str)[:20000]
        result: dict[str, Any] = {}

        def on_verdict(inp: dict[str, Any]) -> str | None:
            result.update(inp)
            return None

        out = run_loop(conv=conv, first_message=msg, tools=tools, ctx=ctx, controls={"verdict": on_verdict},
                       ledger=self.ledger, who=f"verifier@{node.id}", role="verifier", max_turns=20,
                       max_tokens=200_000, wall_seconds=900)
        if out.control != "verdict":
            return []  # a failed review does not block the node; the contract checks already passed
        return [] if result.get("accept") else [f"(LLM review) {r}" for r in result.get("reasons", [])] or ["(LLM review) rejected"]


def _parse_claims(raw: Any) -> list[Claim]:
    out = []
    for c in raw if isinstance(raw, list) else []:
        if isinstance(c, dict) and "key" in c:
            out.append(Claim(key=str(c["key"]), value=c.get("value"), evidence=c.get("evidence")))
    return out
