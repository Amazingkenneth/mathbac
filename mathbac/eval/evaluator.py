"""Runs every checkpoint on a finished run and writes eval/results.json and eval/report.md."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from ..task import Task
from . import comm
from .checkpoints import ALL, CheckResult, EvalContext

REPO_ROOT = Path(__file__).resolve().parents[2]


def evaluate(run_dir: Path, pdb_cache: Path | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())["run"]
    task_path = Path(cfg["task"])
    task = Task.load(task_path if task_path.is_absolute() else REPO_ROOT / task_path)
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    reviews_p = eval_dir / "reviews.yaml"
    reviews = yaml.safe_load(reviews_p.read_text()) if reviews_p.exists() else {}

    ctx = EvalContext(run_dir=run_dir, workspace=run_dir / "workspace", gold_dir=task.gold_dir,
                      pdb_cache=pdb_cache or REPO_ROOT / "cache" / "pdb",
                      access_level=cfg.get("access_level", "L3"), reviews=reviews or {})
    results: list[CheckResult] = []
    for fn in ALL:
        try:
            results.append(fn(ctx))
        except Exception as e:  # one broken checker must not hide the others
            cid = fn.__name__.split("_")[0].upper()
            results.append(CheckResult(cid, fn.__name__, "not_scored", 0.0, [f"checker crashed: {type(e).__name__}: {e}"]))

    cps = task.config.get("checkpoints", {})
    injected = set(cfg.get("injected_nodes", []))
    nodes_of = {cid: c.get("nodes") or [c["node"]] for cid, c in cps.items()}
    total_w = sum(c["weight"] for cid, c in cps.items() if cid != "C7")
    score = 0.0
    rows = []
    for r in results:
        w = cps.get(r.id, {}).get("weight", 0)
        assisted = bool(set(nodes_of.get(r.id, [])) & injected)
        if r.id != "C7":
            score += w * r.score
        rows.append({**asdict(r), "weight": w, "assisted": assisted})
    status = {r.id: r.status for r in results}
    communication = comm.compute(run_dir, task.gold_dir, status, nodes_of)

    out = {
        "run": run_dir.name,
        "access_level": ctx.access_level,
        "plan_mode": cfg.get("plan_mode", "reference"),
        "role_mode": cfg.get("role_mode", "predefined"),
        "reproduction_score": round(100 * score / total_w, 1) if total_w else 0.0,
        "assisted_checkpoints": [r["id"] for r in rows if r["assisted"]],
        "checkpoints": rows,
        "communication": communication,
    }
    (eval_dir / "results.json").write_text(json.dumps(out, indent=1, default=str))
    (eval_dir / "report.md").write_text(render(out))
    return out


def render(out: dict[str, Any]) -> str:
    lines = [
        f"# Evaluation: {out['run']}",
        "",
        f"Access level **{out['access_level']}**, plan mode **{out['plan_mode']}**, roles **{out['role_mode']}**.",
        "",
        f"**Reproduction score: {out['reproduction_score']} / 100**"
        + (f" (assisted: {', '.join(out['assisted_checkpoints'])})" if out["assisted_checkpoints"] else ""),
        "",
        "| Checkpoint | Status | Score | Weight | Notes |",
        "|---|---|---|---|---|",
    ]
    for r in out["checkpoints"]:
        note = r["details"][0] if r["details"] else ""
        lines.append(f"| {r['id']} {r['title']} | {r['status']} | {r['score']:.2f} | {r['weight']} | {note} |")
    lines += ["", "## Details", ""]
    for r in out["checkpoints"]:
        lines.append(f"### {r['id']} {r['title']}")
        lines += [f"- {d}" for d in r["details"]] + [""]
    c = out["communication"]
    sr = c["spec_retention"]
    lines += [
        "## Communication",
        "",
        f"- Fact coverage (facts passed to at least one node): {_pct(sr['fact_coverage'])}; fact accuracy: {_pct(sr['fact_accuracy'])}",
        f"- Facts never passed: {', '.join(sr['facts_never_passed']) or 'none'}",
        f"- Grounding ratio: {_pct(c['grounding_ratio'])}",
        f"- Contradiction rate: {_pct(c['contradiction_rate'])}" + (f" ({', '.join(c['contradicted_keys'])})" if c["contradicted_keys"] else ""),
        f"- Tool error rate: {_pct(c['tool_error_rate'])}",
        f"- Verifier rejections: {c['verifier_rejections'] or 'none'}",
        f"- Undetected faults (accepted in the loop, failed in evaluation): {c['undetected_faults'] or 'none'}",
        f"- Cost: ${c['cost']['usd_total']}" + (" (incomplete: some models have no price)" if c["cost"]["usd_incomplete"] else "")
        + f", {c['cost']['model_turns']} model turns, {c['cost']['wall_seconds'] / 60:.1f} min wall clock",
        "",
    ]
    return "\n".join(lines)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"
