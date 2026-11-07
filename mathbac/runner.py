"""Sets up a run directory and wires the manager, workers, sandbox, queue and gateway together.

Run directory layout::

    runs/<run_id>/
      config.yaml       snapshot of the run and model configs
      ledger.jsonl      every event (see ledger.py)
      dag.json          final task graph with statuses
      summary.json      run outcome
      workspace/        what agents see as /work
      cache/            compute-queue result cache
      eval/             evaluator output (written by `mathbac eval`)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from .agents import ContractVerifier, Manager, RoleBook, RunSummary, WorkerAgent
from .compute import JobQueue
from .dag import TaskDAG
from .gateway import Backend, ModelRouter
from .ledger import Ledger
from .sandbox import Sandbox, make_sandbox
from .schemas import NodeSpec
from .task import Task
from .tools import ToolContext, ToolRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def new_run_id(cfg: dict[str, Any]) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{cfg.get('access_level', 'L3')}-{cfg.get('plan_mode', 'reference')}-{cfg.get('role_mode', 'predefined')}"


def run(
    run_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
    *,
    runs_dir: Path = REPO_ROOT / "runs",
    run_id: str | None = None,
    backends: dict[str, Backend] | None = None,
    sandbox: Sandbox | None = None,
) -> tuple[Path, RunSummary]:
    task = Task.load(REPO_ROOT / run_cfg["task"] if not Path(run_cfg["task"]).is_absolute() else Path(run_cfg["task"]))
    run_dir = Path(runs_dir) / (run_id or new_run_id(run_cfg))
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"run": run_cfg, "models": models_cfg}, sort_keys=False))

    ledger = Ledger(run_dir)
    level = run_cfg.get("access_level", "L3")
    materials_root = Path(run_cfg.get("materials_root", REPO_ROOT / "materials" / task.id))
    if not materials_root.is_absolute():
        materials_root = REPO_ROOT / materials_root
    materials = task.materials_for(level, materials_root)
    wanted = task.config["access_levels"][level]
    ledger.append("materials", access_level=level, root=str(materials_root), mounted=sorted(materials),
                  missing=[m for m in wanted if m not in materials])

    router = ModelRouter(models_cfg, backends=backends)
    secrets = router.secret_values()
    registry = ToolRegistry.load(REPO_ROOT / run_cfg.get("toolcards", "toolcards"))
    sb = sandbox or make_sandbox(run_cfg.get("sandbox", {}), workspace, materials)
    qcfg = run_cfg.get("compute", {})
    queue = JobQueue(sb, ledger, run_dir / "cache", backend=qcfg.get("backend", "local"),
                     max_workers=int(qcfg.get("max_workers", 4)), slurm=qcfg.get("slurm"))
    roles = RoleBook(run_cfg.get("role_mode", "predefined"))
    verifier = ContractVerifier(task, workspace)

    def ctx_factory(node: NodeSpec) -> ToolContext:
        return ToolContext(workspace=workspace, materials=materials, sandbox=sb, queue=queue,
                           registry=registry, ledger=ledger, node=node, secrets=secrets)

    worker = WorkerAgent(router=router, roles=roles, contracts=task.contracts(), verifier=verifier, ledger=ledger,
                         ctx_factory=ctx_factory, access_level=level,
                         llm_review=bool(run_cfg.get("verifier", {}).get("llm_review", False)),
                         max_rejections=int(run_cfg.get("verifier", {}).get("max_rejections", 3)))

    plan_mode = run_cfg.get("plan_mode", "reference")
    dag = task.reference_dag() if plan_mode == "reference" else TaskDAG()
    if plan_mode == "reference" and run_cfg.get("only_nodes"):
        dag = dag.subset(run_cfg["only_nodes"])

    brief = task.brief() + "\n\n# Artifact contracts\n\n" + task.contracts() + \
        "\n\n# Scientific tools\n\n" + registry.summary() + \
        f"\n\n# Access level\n\n{level}: materials available under /materials: {', '.join(sorted(materials)) or 'none'}"
    manager = Manager(dag=dag, run_node=worker.run, ledger=ledger, roles=roles, plan_mode=plan_mode, router=router,
                      task_brief=brief, max_parallel=int(run_cfg.get("max_parallel", 3)),
                      max_attempts=int(run_cfg.get("max_attempts", 2)),
                      max_manager_turns=int(run_cfg.get("max_manager_turns", 80)), secrets=secrets)
    try:
        summary = manager.run()
    finally:
        queue.shutdown()
        if sandbox is None:
            sb.close()
    (run_dir / "dag.json").write_text(json.dumps(dag.to_dict(), indent=1))
    (run_dir / "summary.json").write_text(json.dumps(summary.__dict__, indent=1))
    return run_dir, summary
