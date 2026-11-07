import json
from pathlib import Path

import numpy as np
import yaml

from mathbac.eval import checkpoints as cp
from mathbac.science import pdb
from mathbac.task import Task

REPO = Path(__file__).resolve().parents[1]
TASK = Task.load(REPO / "tasks" / "cha2022")


def ctx(tmp_path, **kw):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return cp.EvalContext(run_dir=tmp_path, workspace=ws, gold_dir=TASK.gold_dir, pdb_cache=tmp_path / "pdb", **kw)


def binormal(target_auc, n=6000, folds=10, seed=0):
    from statistics import NormalDist

    rng = np.random.default_rng(seed)
    d = NormalDist().inv_cdf(target_auc) * np.sqrt(2)
    y = rng.integers(0, 2, n)
    z = rng.normal(size=n) + d * y
    s = 1 / (1 + np.exp(-(z - d / 2)))
    return np.arange(n) % folds, y, s


def test_c4_recomputes_and_checks_orderings(tmp_path):
    c = ctx(tmp_path)
    pred = c.workspace / "ml" / "cv" / "predictions"
    pred.mkdir(parents=True)
    gold = yaml.safe_load((TASK.gold_dir / "table1.yaml").read_text())
    seed = 0
    for model, subsets in gold["table"].items():
        for subset, mets in subsets.items():
            seed += 1
            fold, y, s = binormal(mets["auc"][0], seed=seed)
            np.savez(pred / f"{model}_{subset}.npz", fold=fold, y_true=y, y_score=s)
    r = cp.c4_table1(c)
    aucs = {(row["model"], row["subset"]): row for row in r.data["cells"] if row["metric"] == "auc"}
    assert abs(aucs[("dnn", "GT")]["reproduced"] - 0.877) < 0.02
    assert all(r.data["orderings"].values()), r.data["orderings"]

    # Replace the DNN GT predictions with noise: its cells and the GT > GE ordering must now fail.
    rng = np.random.default_rng(99)
    np.savez(pred / "dnn_GT.npz", fold=np.arange(6000) % 10, y_true=rng.integers(0, 2, 6000), y_score=rng.random(6000))
    r2 = cp.c4_table1(c)
    assert r2.status == "fail"
    assert r2.data["orderings"]["gt_gt_ge"] is False
    assert r2.score < r.score


def test_c4_missing_predictions(tmp_path):
    r = cp.c4_table1(ctx(tmp_path))
    assert r.status == "fail" and r.score == 0.0 and "no predictions" in r.details[-1]


FAKE_PDB = "\n".join(
    [f"ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C" for i, x in enumerate(range(0, 40, 4), 1)]
    + [f"ATOM  {i+20:5d}  CA  GLY B{i:4d}    {x:8.3f}{5.0:8.3f}{0.0:8.3f}  1.00  0.00           C" for i, x in enumerate(range(0, 40, 4), 1)]
)


def test_pdb_parser_and_interface():
    a, b = pdb.parse_ca(FAKE_PDB, "A"), pdb.parse_ca(FAKE_PDB, "B")
    assert len(a.resnums) == 10 and a.label(0) == "ALA1"
    ia, ib = pdb.interface(a, b, 7.0)
    assert ia == {str(i) for i in range(1, 11)}  # every residue is 5 Å from its partner


def test_c5_scores_from_pair_probabilities(tmp_path, monkeypatch):
    monkeypatch.setattr(pdb, "load_chain", lambda pid, ch, cache: pdb.parse_ca(FAKE_PDB, ch))
    c = ctx(tmp_path)
    d = c.workspace / "predictions" / "ppi"
    d.mkdir(parents=True)
    for pid in ("6wzo", "6nqn", "1c29"):
        rows = ["res_a,res_b,prob"] + [f"{i},{j},{0.9 if i == j else 0.01}" for i in range(1, 11) for j in range(1, 11)]
        (d / f"{pid}_dnn_GE_GT.csv").write_text("\n".join(rows))
    r = cp.c5_heldout(c)
    # top 0.1% of 100 pairs = 1 pair -> 1 of 10 true interface residues per chain
    assert r.status == "fail"
    assert all(abs(a - 0.1) < 1e-9 for a in r.data["accuracy"])


def test_c6_residue_and_ranking_findings(tmp_path, monkeypatch):
    chain_text = "\n".join(
        f"ATOM  {k:5d}  CA  {name} A{num:4d}    {0.0:8.3f}{0.0:8.3f}{float(k):8.3f}  1.00  0.00           C"
        for k, (name, num) in enumerate([("TYR", 293), ("ARG", 294), ("ARG", 307), ("TYR", 313), ("ALA", 400)], 1))
    monkeypatch.setattr(pdb, "load_chain", lambda pid, ch, cache: pdb.parse_ca(chain_text, "A"))
    c = ctx(tmp_path, reviews={"gqd_iapp": "pass"})
    (c.workspace / "data" / "np").mkdir(parents=True)
    (c.workspace / "data" / "np" / "cases.csv").write_text("case,np_name,pdb_id,chain,reference\nCNT-1CXP,cnt17,1cxp,A,Kagan\n")
    p = c.workspace / "predictions" / "pni"
    p.mkdir(parents=True)
    (p / "CNT-1CXP_interface.json").write_text(json.dumps({"protein_residues": ["293", "294", "313"], "np_nodes": [1]}))
    (p / "corona_ranking.csv").write_text(
        "np_name,protein,score\nSWNT,albumin,0.2\nSWNT,apolipoprotein,0.6\nSWNT,histidine-rich glycoprotein,0.5\n"
        "SWNT,galectin-3-binding protein,0.7\nspherical_C,albumin,0.1\nspherical_C,apolipoprotein,0.3\n"
        "spherical_C,histidine-rich glycoprotein,0.2\nspherical_C,galectin-3-binding protein,0.4\n")
    r = cp.c6_pni(c)
    f = r.data["findings"]
    assert f["cnt_hmpo"] is True and f["corona"] is True and f["spherical_lower"] is True
    assert f["gqd_iapp"] is True and f["zno_bgal"] is None and f["gqd_psma"] is False
    assert r.status == "needs_review"  # 4 passed, 1 pending, pass needs 5


def test_c1_counts_code_items_only_with_code(tmp_path):
    from .test_run_e2e import GOOD_SPEC

    c = ctx(tmp_path, access_level="L1")
    (c.workspace / "spec").mkdir()
    bad = dict(GOOD_SPEC, contact_cutoff_A={"value": 8.0, "source": "x", "status": "stated"})
    (c.workspace / "spec" / "spec.yaml").write_text(yaml.safe_dump(bad))
    r = cp.c1_spec(c)
    assert "contact_cutoff" in r.data["misses"]
    assert "opd_sizes" not in r.data["hits"] + r.data["misses"]  # code-only item skipped at L1
