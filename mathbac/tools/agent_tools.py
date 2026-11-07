"""Tools exposed to worker agents.

Each tool is a JSON-schema function plus a handler. We define our own
``run_shell`` instead of a provider's built-in bash tool so that every model
behind the gateway (Claude, GPT, OpenRouter models) sees the same tool surface,
which keeps the model-mixing experiment fair.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..compute import JobQueue
from ..gateway import ToolSpec, redact
from ..ledger import Ledger
from ..sandbox import Sandbox, truncate
from ..schemas import JobSpec, NodeSpec
from .registry import ToolRegistry, ToolRegistryError

MAX_READ_CHARS = 40_000


class PathError(ValueError):
    pass


@dataclass
class ToolContext:
    workspace: Path
    materials: dict[str, Path]
    sandbox: Sandbox
    queue: JobQueue
    registry: ToolRegistry
    ledger: Ledger
    node: NodeSpec
    secrets: list[str] = field(default_factory=list)

    # ---- path confinement ---------------------------------------------
    def resolve(self, raw: str, write: bool = False) -> Path:
        """Map an agent-supplied path to a host path inside the workspace or an allowed materials folder.

        Accepts 'rel/path', '/work/rel/path', 'materials/<name>/...', '/materials/<name>/...'.
        Rejects anything that escapes those roots (.., symlinks, other absolute paths).
        """
        p = raw.strip()
        if not p:
            raise PathError("empty path")
        for prefix in ("/work/", "./"):
            if p.startswith(prefix):
                p = p[len(prefix):]
        if p in ("/work", "."):
            p = ""
        if p.startswith("/materials/") or p.startswith("materials/"):
            rest = p.split("materials/", 1)[1]
            name, _, sub = rest.partition("/")
            if name not in self.materials:
                raise PathError(f"materials/{name} is not available at this access level")
            if write:
                raise PathError("materials are read-only")
            root = self.materials[name].resolve()
            target = (root / sub).resolve()
            if not target.is_relative_to(root):
                raise PathError(f"path escapes materials/{name}")
            return target
        if p.startswith("/"):
            raise PathError("absolute paths must start with /work/ or /materials/")
        root = self.workspace.resolve()
        target = (root / p).resolve()
        if not target.is_relative_to(root):
            raise PathError("path escapes the workspace")
        return target


Handler = Callable[[ToolContext, dict[str, Any]], tuple[str, bool]]


@dataclass
class AgentTool:
    spec: ToolSpec
    handler: Handler | None  # None for control tools handled by the worker loop (finish)


def _schema(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


# ---- handlers ------------------------------------------------------------

def _read_file(ctx: ToolContext, a: dict) -> tuple[str, bool]:
    path = ctx.resolve(a["path"])
    if not path.is_file():
        return f"not a file: {a['path']}", True
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        return f"{a['path']} is binary ({len(data)} bytes). Inspect it with run_shell (e.g. python, h5ls, pdftotext).", True
    lines = data.decode(errors="replace").splitlines()
    start = max(int(a.get("start_line", 1)), 1)
    count = int(a.get("max_lines", 400))
    chunk = lines[start - 1 : start - 1 + count]
    body = "\n".join(f"{i:>6}  {line}" for i, line in enumerate(chunk, start))
    more = f"\n[{len(lines) - (start - 1 + len(chunk))} more lines]" if start - 1 + len(chunk) < len(lines) else ""
    return truncate(body, MAX_READ_CHARS) + more, False


def _write_file(ctx: ToolContext, a: dict) -> tuple[str, bool]:
    path = ctx.resolve(a["path"], write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(a["content"])
    ctx.ledger.record_artifact(ctx.workspace, str(path.relative_to(ctx.workspace.resolve())), ctx.node.id)
    return f"wrote {len(a['content'])} characters to {a['path']}", False


def _list_dir(ctx: ToolContext, a: dict) -> tuple[str, bool]:
    raw = a.get("path", ".").strip()
    if raw.rstrip("/") in ("/materials", "materials", "/work/materials"):
        return "\n".join(f"{name}/" for name in sorted(ctx.materials)) or "(no materials at this access level)", False
    path = ctx.resolve(raw)
    if not path.is_dir():
        return f"not a directory: {a.get('path')}", True
    entries = []
    for p in sorted(path.iterdir())[:500]:
        kind = "/" if p.is_dir() else f"  {p.stat().st_size} B"
        entries.append(f"{p.name}{kind}")
    return "\n".join(entries) or "(empty)", False


def _run_shell(ctx: ToolContext, a: dict) -> tuple[str, bool]:
    timeout = min(float(a.get("timeout_s", 300)), 1800.0)
    res = ctx.sandbox.exec(a["command"], timeout_s=timeout)
    head = f"exit code {res.exit_code} after {res.seconds:.1f}s" + (" (timed out)" if res.timed_out else "")
    return f"{head}\n{res.output}", res.exit_code != 0


def _list_tools(ctx: ToolContext, a: dict) -> tuple[str, bool]:
    return ctx.registry.summary(ctx.node.tools or None), False


def _read_tool_card(ctx: ToolContext, a: dict) -> tuple[str, bool]:
    return ctx.registry.render(a["name"]), False


def _run_tool(ctx: ToolContext, a: dict) -> tuple[str, bool]:
    name = a["name"]
    if ctx.node.tools and name not in ctx.node.tools:
        return f"{name} is not in this node's tool list ({', '.join(ctx.node.tools)})", True
    args = a.get("args") or {}
    command = ctx.registry.command(name, args)
    card = ctx.registry.get(name)
    res = card.resources or {}
    inputs = [str(args[k]) for k, arg in card.args.items() if arg.input and isinstance(args.get(k), str)]
    spec = JobSpec(
        tool=name, args=args, node_id=ctx.node.id, input_paths=inputs,
        cpus=int(res.get("cpus", 1)), gpus=int(res.get("gpus", 0)),
        mem_gb=float(res.get("mem_gb", 4)), time_limit_min=float(res.get("time_limit_min", 60)),
    )
    rec = ctx.queue.submit(spec, command)
    if a.get("wait", True) and rec.state != "cached":
        rec = ctx.queue.wait(rec.job_id, timeout_s=float(a.get("wait_timeout_s", 1800)))
    return json.dumps({"job_id": rec.job_id, "state": rec.state, "exit_code": rec.exit_code,
                       "output_tail": rec.output_tail[-3000:]}, indent=1), rec.state == "failed"


def _job_status(ctx: ToolContext, a: dict) -> tuple[str, bool]:
    wait_s = float(a.get("wait_s") or 0)
    rec = ctx.queue.wait(a["job_id"], timeout_s=wait_s) if wait_s > 0 else ctx.queue.status(a["job_id"])
    return json.dumps({"job_id": rec.job_id, "state": rec.state, "exit_code": rec.exit_code,
                       "output_tail": rec.output_tail[-3000:]}, indent=1), rec.state == "failed"


_CLAIMS_SCHEMA = {
    "type": "array",
    "description": "Facts you established, each tied to evidence. Use the fact keys from your node brief where they apply.",
    "items": _schema(
        {
            "key": {"type": "string"},
            "value": {"description": "Number, string, boolean or list"},
            "evidence": {"type": "string", "description": "Workspace path (optionally path#anchor) that supports the claim"},
        },
        ["key", "value", "evidence"],
    ),
}


def worker_tools() -> list[AgentTool]:
    return [
        AgentTool(ToolSpec("read_file", "Read a text file from the workspace or /materials. Returns numbered lines.",
                           _schema({"path": {"type": "string"},
                                    "start_line": {"type": "integer", "description": "1-based first line (default 1)"},
                                    "max_lines": {"type": "integer", "description": "Default 400"}}, ["path"])), _read_file),
        AgentTool(ToolSpec("write_file", "Create or overwrite a text file in the workspace. Records the artifact hash.",
                           _schema({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"])), _write_file),
        AgentTool(ToolSpec("list_dir", "List a directory in the workspace or /materials.",
                           _schema({"path": {"type": "string", "description": "Default: workspace root"}}, [])), _list_dir),
        AgentTool(ToolSpec("run_shell",
                           "Run a bash command in the sandbox (cwd = workspace). Use for Python scripts, conversions and quick checks. "
                           "Heavy computations go through run_tool instead.",
                           _schema({"command": {"type": "string"},
                                    "timeout_s": {"type": "number", "description": "Default 300, max 1800"}}, ["command"])), _run_shell),
        AgentTool(ToolSpec("list_tools", "List the scientific tools available to this node.", _schema({}, [])), _list_tools),
        AgentTool(ToolSpec("read_tool_card", "Read a tool's instruction card. Always read it before the first run_tool call for that tool.",
                           _schema({"name": {"type": "string"}}, ["name"])), _read_tool_card),
        AgentTool(ToolSpec("run_tool",
                           "Submit a scientific tool run to the compute queue. Results are cached by input. "
                           "With wait=true (default) the call blocks until the job ends.",
                           _schema({"name": {"type": "string"}, "args": {"type": "object"},
                                    "wait": {"type": "boolean"},
                                    "wait_timeout_s": {"type": "number"}}, ["name", "args"])), _run_tool),
        AgentTool(ToolSpec("job_status", "Check a queued job, optionally waiting up to wait_s seconds.",
                           _schema({"job_id": {"type": "string"}, "wait_s": {"type": "number"}}, ["job_id"])), _job_status),
        AgentTool(ToolSpec("finish",
                           "Call once when the node's outputs are written. The verifier then checks them against the acceptance test; "
                           "if it fails you get the reasons back and can keep working.",
                           _schema({"summary": {"type": "string", "description": "What you did and anything the manager must know"},
                                    "outputs": {"type": "array", "items": {"type": "string"}, "description": "Workspace paths you produced"},
                                    "claims": _CLAIMS_SCHEMA}, ["summary", "outputs", "claims"])), None),
    ]


def run_handler(tool: AgentTool, ctx: ToolContext, args: dict[str, Any]) -> tuple[str, bool, float]:
    t0 = time.time()
    try:
        out, err = tool.handler(ctx, args)  # type: ignore[misc]
    except (PathError, ToolRegistryError, KeyError, ValueError) as e:
        out, err = f"{type(e).__name__}: {e}", True
    except OSError as e:
        out, err = f"OS error: {e}", True
    return redact(out, ctx.secrets), err, time.time() - t0
