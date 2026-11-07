"""End-to-end runs of the harness with scripted models and the local sandbox."""

from pathlib import Path

import yaml

from mathbac.eval import evaluate
from mathbac.ledger import read_events
from mathbac.runner import run

from .fakes import call, fake_router, turn

REPO = Path(__file__).resolve().parents[1]


def v(value, source="paper p.247 Methods", status="stated"):
    return {"value": value, "source": source, "status": status}


GOOD_SPEC = {
    "residue_representation": v("CA"),
    "contact_cutoff_A": v(7.0),
    "graph_edge_cutoff_A": v(7.0),
    "descriptors": v({
        "CH": ["electrostatic charge", "hydrophobicity", "molecular weight", "polarity", "atom counts"],
        "GE": ["minimum inaccessible radius", "accessible shell volume", "pocketness", "OPD chirality"],
        "GT": ["GNM", "Ollivier-Ricci", "Forman-Ricci", "multifractal dimension"],
    }),
    "opd_neighbourhood_sizes": v([5, 7, 10, 15], "code: local_OPD.py:93", "code_only"),
    "gnm_modes_fraction": v({"paper": 0.1, "code": "first 10 modes"}, "Methods; GNM call in PPI_descriptors_cal.py", "conflict"),
    "node_curvature_aggregation": v("sum of edge curvatures"),
    "mfd_features": v(["fd", "more_fd_1", "more_fd_2", "more_fd_3", "more_fd_4"], "code: MFD.py", "code_only"),
    "training_set": v({"n_pairs": 464, "n_residue_pairs": 27859297}),
    "class_balancing": v("undersampling of non-contacts (binary_under)", "code", "code_only"),
    "descriptor_subsets": v(["all", "CH", "GE", "GT", "GE+GT", "CH+GE", "CH+GT"]),
    "models": v(["logistic regression", "Gaussian naive Bayes", "SVM", "random forest", "XGBoost", "DNN"]),
    "dnn": v({"layers": 3, "units": 512, "activation": "relu", "dropout": 0.5, "output": "softmax", "loss": "categorical cross-entropy"}),
    "cv_folds": v(10),
    "metrics": v(["accuracy", "ROC-AUC", "F1"]),
    "interface_threshold_top_fraction": v(0.001),
    "heldout_ppi": v(["6WZO", "6NQN", "1C29"]),
    "np_coarse_graining": v({"carbon_atoms_per_node": 13, "zno_atoms_per_node": 2, "np_graph_edge_cutoff_A": 7.0}),
    "np_descriptor_subset": v("GE+GT"),
    "extra": v({}, "n/a", "unstated"),
}


def spec_reader_policy():
    """Calls finish too early once (verifier rejects), then writes the spec and finishes."""
    steps = [
        turn(call("list_dir", path="/materials")),
        turn(call("finish", summary="done?", outputs=[], claims=[])),
        turn(call("write_file", path="spec/spec.yaml", content=yaml.safe_dump(GOOD_SPEC))),
        turn(call("finish", summary="Spec written; GNM modes conflict recorded.", outputs=["spec/spec.yaml"],
                  claims=[{"key": "contact_cutoff_A", "value": 7.0, "evidence": "spec/spec.yaml#contact_cutoff_A"},
                          {"key": "n_pairs", "value": 464, "evidence": "spec/spec.yaml#training_set"},
                          {"key": "made_up", "value": 1, "evidence": "spec/nothing_here.yaml"}])),
    ]
    return lambda conv: steps[conv.step] if conv.step < len(steps) else turn(text="done")


def base_cfg(tmp_path, **kw):
    mat = tmp_path / "materials"
    for d in ("paper", "si", "code"):
        (mat / d).mkdir(parents=True, exist_ok=True)
    (mat / "paper" / "paper.txt").write_text("paper text")
    return {"task": str(REPO / "tasks" / "cha2022"), "access_level": "L2", "materials_root": str(mat),
            "toolcards": str(REPO / "toolcards"), "sandbox": {"kind": "local_unsafe"}, "max_parallel": 2, **kw}


def test_reference_mode_single_node_with_rejection_then_eval(tmp_path):
    models, backends = fake_router({"spec_reader": spec_reader_policy()})
    run_dir, summary = run(base_cfg(tmp_path, plan_mode="reference", only_nodes=["N1"]), models,
                           runs_dir=tmp_path / "runs", run_id="t1", backends=backends)
    assert summary.status == "complete", summary
    events = list(read_events(run_dir))
    kinds = [e["kind"] for e in events if e["event"] == "message"]
    assert kinds.count("rejection") == 1 and kinds.count("result") == 2 and kinds.count("spec") == 1

    # the worker saw the materials listing and the verifier's reasons
    conv = backends["fake_spec_reader"].conversations[0]
    assert "paper" in conv.inbox[1][0].content
    assert "was not written" in conv.inbox[2][0].content

    out = evaluate(run_dir, pdb_cache=tmp_path / "pdb")
    c1 = next(c for c in out["checkpoints"] if c["id"] == "C1")
    assert c1["status"] == "pass", c1["details"]
    assert c1["score"] == 1.0
    comm = out["communication"]
    assert comm["verifier_rejections"] == {"N1": 1}
    assert abs(comm["grounding_ratio"] - 2 / 3) < 1e-9  # one claim points at a file that does not exist
    assert (run_dir / "eval" / "report.md").exists()
    # gold data never lands in the workspace
    assert not any("spec_checklist" in p.name for p in (run_dir / "workspace").rglob("*"))


def test_llm_manager_plans_dispatches_and_finishes(tmp_path):
    node = {"id": "S1", "role": "spec_reader", "goal": "Write spec/spec.yaml", "outputs": [{"path": "spec/spec.yaml"}],
            "acceptance": "spec_contract", "facts": {"contact_cutoff": 7.0, "cv_folds": 5}}

    def manager(conv):
        if conv.step == 0:
            return turn(call("add_node", node={**node, "role": "nonexistent"}), call("add_node", node=node))
        if conv.step == 1:
            assert "unknown role" in conv.inbox[-1][0].content
            return turn(text="Dispatching S1.")
        assert "S1 done" in conv.inbox[-1]
        return turn(call("finish_run", summary="spec extracted"))

    models, backends = fake_router({"manager": manager, "spec_reader": spec_reader_policy()})
    run_dir, summary = run(base_cfg(tmp_path, plan_mode="llm"), models, runs_dir=tmp_path / "runs", run_id="t2", backends=backends)
    assert summary.status == "complete", summary
    assert summary.summary == "spec extracted"
    out = evaluate(run_dir, pdb_cache=tmp_path / "pdb")
    sr = out["communication"]["spec_retention"]
    # the manager passed contact_cutoff (correct, via alias) and cv_folds (wrong: 5, not 10)
    assert sr["facts_passed"]["contact_cutoff_A"] == "1/1 correct"
    assert sr["facts_passed"]["cv_folds"] == "0/1 correct"
    assert sr["fact_accuracy"] == 0.5


def test_dynamic_roles_require_definition(tmp_path):
    node = {"id": "S1", "role": "paper_parser", "goal": "Write spec/spec.yaml", "outputs": [{"path": "spec/spec.yaml"}]}

    def manager(conv):
        if conv.step == 0:
            return turn(call("define_role", name="paper_parser", prompt="Role: Paper Parser. Extract the spec."),
                        call("add_node", node=node))
        if conv.step == 1:
            return turn(text="go")
        return turn(call("finish_run", summary="ok"))

    models, backends = fake_router({"manager": manager, "paper_parser": spec_reader_policy()})
    run_dir, summary = run(base_cfg(tmp_path, plan_mode="llm", role_mode="dynamic"), models,
                           runs_dir=tmp_path / "runs", run_id="t3", backends=backends)
    assert summary.status == "complete", summary
    worker_conv = backends["fake_paper_parser"].conversations[0]
    assert "Role: Paper Parser" in worker_conv.system
    assert any(e["event"] == "dag_edit" and e["op"] == "define_role" for e in read_events(run_dir))
