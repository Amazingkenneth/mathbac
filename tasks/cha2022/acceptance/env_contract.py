"""N2: env/smoke_report.json lists every tool the node was given, with a valid status."""

from pathlib import Path

from mathbac.contracts import load_json


def check(workspace: Path, node) -> list[str]:
    try:
        rep = load_json(workspace / "env" / "smoke_report.json")
    except Exception as e:
        return [f"env/smoke_report.json is not valid JSON: {e}"]
    tools = {t.get("name"): t for t in rep.get("tools", []) if isinstance(t, dict)}
    problems = [f"no smoke-test entry for {name}" for name in node.tools if name not in tools]
    for name, t in tools.items():
        if t.get("status") not in ("ok", "failed", "missing"):
            problems.append(f"{name}: status must be ok, failed or missing")
        if t.get("status") == "ok" and not t.get("command"):
            problems.append(f"{name}: an ok status needs the command that was run")
    return problems
