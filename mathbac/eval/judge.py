"""LLM judge for C7 (report rubric) and the C6 findings that need review.

The judge sees the report, the claim verdicts, the evaluator's own results and the
relevant prediction files. Its scores are written to eval/c7_scores.yaml and
eval/reviews.yaml, marked `by: llm-judge`, so a person can overwrite them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ..gateway import ModelRouter, ToolSpec
from .checkpoints import RUBRIC

SYSTEM = """You review a reproduction of a computational science paper. You are given the team's report, \
its claim-by-claim verdicts, and an independent evaluator's recomputed results. Score the report from 1 to 5 on each rubric \
dimension and judge each listed finding. Base every judgement on the evidence given; when the evidence is missing, \
score low and say so. Call the `scores` tool exactly once."""

RUBRIC_TEXT = {
    "fidelity": "Does the workflow follow the published method, with deviations stated?",
    "execution": "Did the steps actually run, end to end, from the team's own code?",
    "accuracy": "Do the recomputed numbers match the paper?",
    "completeness": "Are all claims addressed with evidence?",
    "rigor": "Are verdicts justified, uncertainty acknowledged, and nothing overstated?",
}


def judge(run_dir: Path, models_cfg: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
    run_dir = Path(run_dir)
    ws = run_dir / "workspace"
    results = json.loads((run_dir / "eval" / "results.json").read_text())

    def read(p: Path, limit: int = 30000) -> str:
        return p.read_text()[:limit] if p.exists() else "(missing)"

    review_items = [f for f in findings if f.get("kind") == "review"]
    material = [
        "## Rubric\n" + "\n".join(f"- {k}: {RUBRIC_TEXT[k]}" for k in RUBRIC),
        "## Findings to judge\n" + "\n".join(f"- {f['id']}: {f['desc']}" for f in review_items),
        "## Report\n" + read(ws / "report" / "report.md"),
        "## Claim verdicts\n" + read(ws / "report" / "claims.yaml"),
        "## Evaluator results\n" + json.dumps([{k: r[k] for k in ("id", "status", "score", "details")} for r in results["checkpoints"]], indent=1),
    ]
    for f in review_items:
        for p in sorted((ws / "predictions" / "pni").glob("*_interface.json")):
            if any(m.lower() in p.name.lower() for m in f.get("case_match", [])):
                material.append(f"## {p.name}\n" + read(p, 5000))

    tool = ToolSpec("scores", "Record rubric scores and finding verdicts.", {
        "type": "object", "additionalProperties": False, "required": ["rubric", "findings", "notes"],
        "properties": {
            "rubric": {"type": "object", "properties": {k: {"type": "integer", "minimum": 1, "maximum": 5} for k in RUBRIC},
                       "required": RUBRIC, "additionalProperties": False},
            "findings": {"type": "object", "additionalProperties": {"type": "string", "enum": ["pass", "fail"]}},
            "notes": {"type": "string"},
        },
    })
    router = ModelRouter(models_cfg)
    conv = router.start("judge", system=SYSTEM, tools=[tool])
    turn = conv.send_user("\n\n".join(material) + "\n\nCall the scores tool now.")
    call = next((c for c in turn.tool_calls if c.name == "scores"), None)
    if call is None:
        turn = conv.send_user("Please call the scores tool with your scores.")
        call = next((c for c in turn.tool_calls if c.name == "scores"), None)
    if call is None:
        raise RuntimeError("judge did not return scores")
    out = call.input
    eval_dir = run_dir / "eval"
    (eval_dir / "c7_scores.yaml").write_text(yaml.safe_dump({**out["rubric"], "by": "llm-judge", "notes": out.get("notes", "")}))
    existing = yaml.safe_load((eval_dir / "reviews.yaml").read_text()) if (eval_dir / "reviews.yaml").exists() else {}
    merged = {**{k: v for k, v in out.get("findings", {}).items() if k in {f["id"] for f in review_items}}, **(existing or {})}
    (eval_dir / "reviews.yaml").write_text(yaml.safe_dump(merged))  # a person's earlier review wins over the judge
    return out
