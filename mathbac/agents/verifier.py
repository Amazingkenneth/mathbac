"""In-loop verification of a node's outputs before the manager marks it done.

Two layers:
1. Contract checks (always): every declared output path exists, then the task's
   acceptance check for the node runs on the files.
2. LLM review (optional, ``verifier.llm_review: true``): a verifier agent reads
   the outputs and accepts or rejects them with reasons.

This is separate from the evaluator, which runs after the run with gold data the
agents never see.
"""

from __future__ import annotations

import glob
from pathlib import Path

from ..schemas import NodeSpec
from ..task import Task


def missing_outputs(workspace: Path, node: NodeSpec) -> list[str]:
    problems = []
    for out in node.outputs:
        matches = glob.glob(str(Path(workspace) / out.path), recursive=True)
        if not matches:
            problems.append(f"declared output {out.path!r} was not written")
    return problems


class ContractVerifier:
    def __init__(self, task: Task, workspace: Path):
        self.task = task
        self.workspace = Path(workspace)

    def check(self, node: NodeSpec) -> list[str]:
        problems = missing_outputs(self.workspace, node)
        if problems:
            return problems
        if node.acceptance:
            try:
                fn = self.task.acceptance(node.acceptance)
            except FileNotFoundError as e:
                return [str(e)]
            try:
                problems = list(fn(self.workspace, node))
            except Exception as e:  # an acceptance check crashing is itself a failed check
                problems = [f"acceptance check {node.acceptance} raised {type(e).__name__}: {e}"]
        return problems
