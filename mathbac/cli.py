"""Command line: `mathbac run | eval | judge | dag | tools`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def cmd_run(a: argparse.Namespace) -> int:
    from .runner import run

    cfg = _load(a.config)
    for k in ("access_level", "plan_mode", "role_mode"):
        if getattr(a, k):
            cfg[k] = getattr(a, k)
    if a.only:
        cfg["only_nodes"] = a.only.split(",")
    run_dir, summary = run(cfg, _load(a.models), run_id=a.run_id)
    print(f"run directory: {run_dir}")
    print(json.dumps(summary.__dict__, indent=1))
    if a.eval:
        return cmd_eval(argparse.Namespace(run_dir=str(run_dir)))
    return 0 if summary.status == "complete" else 1


def cmd_eval(a: argparse.Namespace) -> int:
    from .eval import evaluate

    out = evaluate(Path(a.run_dir))
    print((Path(a.run_dir) / "eval" / "report.md").read_text())
    return 0 if out else 1


def cmd_judge(a: argparse.Namespace) -> int:
    from .eval import evaluate
    from .eval.judge import judge
    from .task import Task

    run_dir = Path(a.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    task = Task.load(REPO_ROOT / cfg["run"]["task"])
    if not (run_dir / "eval" / "results.json").exists():
        evaluate(run_dir)
    findings = yaml.safe_load((task.gold_dir / "pni_findings.yaml").read_text())["findings"]
    out = judge(run_dir, _load(a.models), findings)
    print(json.dumps(out, indent=1))
    evaluate(run_dir)  # rescore with the judge's reviews
    print(f"updated {run_dir / 'eval' / 'report.md'}")
    return 0


def cmd_dag(a: argparse.Namespace) -> int:
    from .task import Task

    print(Task.load(Path(a.task)).reference_dag().render())
    return 0


def cmd_tools(a: argparse.Namespace) -> int:
    from .tools import ToolRegistry

    reg = ToolRegistry.load(Path(a.cards))
    print(reg.render(a.name) if a.name else reg.summary())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mathbac", description="Multi-agent paper reproduction (DARPA MATHBAC)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the multi-agent system on a task")
    r.add_argument("--config", default=str(REPO_ROOT / "configs" / "run_cha2022.yaml"))
    r.add_argument("--models", default=str(REPO_ROOT / "configs" / "models.yaml"))
    r.add_argument("--access-level", dest="access_level", choices=["L1", "L2", "L3"])
    r.add_argument("--plan-mode", dest="plan_mode", choices=["reference", "llm"])
    r.add_argument("--role-mode", dest="role_mode", choices=["predefined", "dynamic"])
    r.add_argument("--only", help="comma-separated reference-DAG node ids to run (others' outputs must already exist)")
    r.add_argument("--run-id")
    r.add_argument("--eval", action="store_true", help="evaluate the run when it finishes")
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("eval", help="score a finished run against the checkpoints")
    e.add_argument("run_dir")
    e.set_defaults(fn=cmd_eval)

    j = sub.add_parser("judge", help="LLM-judge the C7 rubric and the C6 review findings, then rescore")
    j.add_argument("run_dir")
    j.add_argument("--models", default=str(REPO_ROOT / "configs" / "models.yaml"))
    j.set_defaults(fn=cmd_judge)

    d = sub.add_parser("dag", help="print a task's reference DAG")
    d.add_argument("--task", default=str(REPO_ROOT / "tasks" / "cha2022"))
    d.set_defaults(fn=cmd_dag)

    t = sub.add_parser("tools", help="list tool cards, or show one")
    t.add_argument("name", nargs="?")
    t.add_argument("--cards", default=str(REPO_ROOT / "toolcards"))
    t.set_defaults(fn=cmd_tools)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
