"""All 14 reference stages with scripted workers writing synthetic artifacts, then the evaluator.

Checks that the contracts, the in-loop acceptance checks and the checkpoints agree on file
layout, and that the manager runs the whole graph (parallel frontier included).
"""

import re
import sys
from pathlib import Path

from mathbac.eval import evaluate
from mathbac.ledger import read_events
from mathbac.runner import run
from mathbac.science import pdb

from . import synthetic
from .fakes import call, fake_router, turn
from .test_run_e2e import base_cfg

REPO = Path(__file__).resolve().parents[1]
ROLES = ["spec_reader", "env_engineer", "data_curator", "descriptor_engineer", "ml_engineer", "reporter"]


def synthetic_worker(conv):
    node = re.match(r"Node (\S+)", conv.inbox[0]).group(1)
    if conv.step == 0:
        cmd = f"cd {REPO} && {sys.executable} tests/synthetic.py {node} \"$WORK\""
        return turn(call("run_shell", command=cmd))
    if conv.step == 1:
        assert not conv.inbox[-1][0].is_error, conv.inbox[-1][0].content
        return turn(call("finish", summary=f"{node} synthetic outputs", outputs=[],
                         claims=[{"key": "node", "value": node, "evidence": "data/pairs.csv" if node != "N1" else "spec/spec.yaml"}]))
    raise AssertionError(f"{node} was rejected: {conv.inbox[-1][0].content}")


def test_whole_reference_graph(tmp_path, monkeypatch):
    models, backends = fake_router({r: synthetic_worker for r in ROLES})
    run_dir, summary = run(base_cfg(tmp_path, plan_mode="reference", max_parallel=4), models,
                           runs_dir=tmp_path / "runs", run_id="full", backends=backends)
    assert summary.status == "complete", summary
    assert set(summary.nodes.values()) == {"done"} and len(summary.nodes) == 14

    # the frontier ran in parallel: at some point more than one node was running
    running, peak = set(), 0
    for e in read_events(run_dir, "node_status"):
        (running.add if e["status"] == "running" else running.discard)(e["node_id"])
        peak = max(peak, len(running))
    assert peak >= 2

    text = synthetic.fake_pdb_text()
    monkeypatch.setattr(pdb, "load_chain", lambda pid, ch, cache: pdb.parse_ca(text, ch))
    out = evaluate(run_dir, pdb_cache=tmp_path / "pdb")
    by_id = {c["id"]: c for c in out["checkpoints"]}
    assert not any("checker crashed" in d for c in out["checkpoints"] for d in c["details"]), by_id
    assert by_id["C0"]["status"] == "pass"
    assert by_id["C1"]["status"] == "pass"
    assert by_id["C2"]["data"]["checks"]["distance_1ma9"] is True
    assert by_id["C2"]["status"] == "fail"  # 2 synthetic pairs, not 464
    assert by_id["C3"]["status"] == "fail"  # synthetic values do not match the authors' 1MA9 descriptors
    assert by_id["C4"]["status"] == "fail" and by_id["C4"]["score"] > 0
    assert len([a for a in by_id["C5"]["data"]["f1"] if a is not None]) == 3
    assert by_id["C6"]["data"]["findings"]["corona"] is True
    assert by_id["C7"]["status"] == "needs_review"
    assert 0 < out["reproduction_score"] < 100
    assert out["communication"]["messages"]["spec"] == 14
    assert out["communication"]["spec_retention"]["fact_accuracy"] == 1.0  # the reference DAG passes correct facts
    assert out["communication"]["contradicted_keys"] == []  # per-node keys such as descriptor_family are not contradictions
    assert out["role_mode"] == "predefined"
