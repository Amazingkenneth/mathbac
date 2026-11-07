"""The paper must pass its own checkpoints.

The authors' published predictions (Fig. 4 and Fig. 5 source data, in the gold folder) are written
into a workspace in the artifact-contract format and scored. If one of these fails, the checkpoint
rule is wrong, not the paper. Needs RCSB access for ground truth; skipped offline.
"""

import csv
import json
import urllib.request
from pathlib import Path

import pytest
import yaml

from mathbac.eval import checkpoints as cp
from mathbac.task import Task

REPO = Path(__file__).resolve().parents[1]
TASK = Task.load(REPO / "tasks" / "cha2022")
GOLD = TASK.gold_dir
PDB_CACHE = REPO / "cache" / "pdb"


def online() -> bool:
    try:
        urllib.request.urlopen("https://files.rcsb.org/download/1MA9.pdb", timeout=10).close()
        return True
    except OSError:
        return (PDB_CACHE / "6WZO.pdb").exists()


pytestmark = pytest.mark.skipif(not online(), reason="needs RCSB or a populated cache/pdb")


def ctx(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return cp.EvalContext(run_dir=tmp_path, workspace=ws, gold_dir=GOLD, pdb_cache=PDB_CACHE)


def test_authors_heldout_predictions_pass_c5(tmp_path):
    c = ctx(tmp_path)
    authors = yaml.safe_load((GOLD / "heldout_authors.yaml").read_text())
    d = c.workspace / "predictions" / "ppi"
    d.mkdir(parents=True)
    for pid in ("6WZO", "6NQN", "1C29"):
        p = authors[pid]["dnn_GE_GT_predicted"]
        (d / f"{pid.lower()}_dnn_GE_GT_interface.json").write_text(json.dumps({"chain_a": p["A"], "chain_b": p["B"], "top_fraction": 0.001}))
    r = cp.c5_heldout(c)
    assert r.status == "pass", r.details
    assert [round(x, 3) for x in r.data["f1"]] == [0.533, 0.400, 0.366]
    assert "Jaccard) 1.00" in r.details[0]


def test_authors_pni_predictions_pass_c6(tmp_path):
    c = ctx(tmp_path)
    authors = yaml.safe_load((GOLD / "pni_authors.yaml").read_text())
    np_dir = c.workspace / "data" / "np"
    np_dir.mkdir(parents=True)
    with (np_dir / "cases.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "np_name", "pdb_id", "chain", "reference"])
        w.writerow(["CNT-1CXP", "swnt_14_0_17A", "1cxp", "A", "Fig. 5d/e"])
        w.writerow(["GQD-PSMa", "gqd_g3", "5i55", "A", "Fig. 5a"])
    p = c.workspace / "predictions" / "pni"
    p.mkdir(parents=True)
    for case in ("CNT-1CXP", "GQD-PSMa"):
        (p / f"{case}_interface.json").write_text(json.dumps({"protein_residues": authors[case]["protein_residues"], "np_nodes": []}))
    with (p / "corona_ranking.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["np_name", "protein", "score"])
        for row in authors["corona"]:
            name = "spherical_5nm" if "sphere" in row["np"] else f"SWNT_{row['np']}"
            prot = {"Galectin 3 binding protein": "galectin-3-binding protein"}.get(row["protein"], row["protein"])
            w.writerow([name, prot, row["probability"]])
    r = cp.c6_pni(c)
    f = r.data["findings"]
    assert f["cnt_hmpo"] is True and f["gqd_psma"] is True and f["corona"] is True and f["spherical_lower"] is True, r.details
    assert r.status == "needs_review"  # the two review findings are pending


def test_authors_1ma9_descriptors_pass_c3_against_themselves(tmp_path):
    ctx_ = ctx(tmp_path)
    families = {"CH": ["c_charge", "hydrophobicity", "mw"], "GE": ["rinacc", "shell", "pocketness", "opd_7"], "GT": ["orc", "frc", "gnm", "mfd"]}
    for ref in sorted((GOLD / "reference_descriptors").glob("*.csv")):
        rows = list(csv.DictReader(ref.open()))
        have = set(rows[0])
        for fam, wanted in families.items():
            cols = [k for k in wanted if k in have]
            out = ctx_.workspace / "features" / fam / ref.name
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["resnum", "resname", *cols])
                for r in rows:
                    w.writerow([r["resnum"], r["resname"], *[r[k] for k in cols]])
    r = cp.c3_descriptors(ctx_)
    assert r.data["columns"] and all(v is not None and v > 0.999 for v in r.data["columns"].values()), r.details
