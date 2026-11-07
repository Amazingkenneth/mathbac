"""Writes small, contract-conforming artifacts for each reference-DAG node.

Used by the full-pipeline test: scripted workers call
``python3 tests/synthetic.py <node_id> <workspace>`` through run_shell.
The data are fake; the point is that every contract, acceptance check and
checkpoint agree on file layout.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

N_RES = 10
PAIRS = [("1ma9", "A", "B", "train"), ("2abc", "A", "B", "train"),
         ("6wzo", "A", "B", "heldout"), ("6nqn", "A", "B", "heldout"), ("1c29", "A", "B", "heldout")]
SUBSETS = ["all", "CH", "GE", "GT", "GE_GT", "CH_GE", "CH_GT"]
WIDTH = {"CH": 4, "GE": 7, "GT": 8}
CASES = [("CNT-1CXP", "cnt17", "1cxp", "A"), ("GQD-5KHB", "gqd", "5khb", "A"), ("GQD-2L86", "gqd", "2l86", "A"),
         ("ZnO-bgal", "zno", "1jz7", "A"), ("SWNT-corona", "swnt", "-", "-"), ("spherical_C-corona", "spherical_C", "-", "-")]


def chain_coords(chain: str) -> np.ndarray:
    y = 0.0 if chain == "A" else 5.0
    return np.array([[4.0 * i, y, 0.0] for i in range(N_RES)])


def fake_pdb_text() -> str:
    """Matching structure for the evaluator's RCSB lookups (monkeypatched in the test)."""
    lines = []
    k = 1
    for ch in ("A", "B"):
        for i, (x, y, z) in enumerate(chain_coords(ch), 1):
            name = ("TYR", "ARG", "ALA")[i % 3]
            lines.append(f"ATOM  {k:5d}  CA  {name} {ch}{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
            k += 1
    return "\n".join(lines)


def wcsv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def spec() -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.test_run_e2e import GOOD_SPEC

    return GOOD_SPEC


def write(node: str, ws: Path) -> None:
    rng = np.random.default_rng(abs(hash(node)) % 2**32)
    if node == "N1":
        (ws / "spec").mkdir(parents=True, exist_ok=True)
        (ws / "spec" / "spec.yaml").write_text(yaml.safe_dump(spec()))
    elif node == "N2":
        tools = ["ghecom", "unified_ppi_descriptors", "gnm_prody", "ricci_curvature", "multifractal_dimension", "opd_chirality", "dnn_train"]
        (ws / "env").mkdir(parents=True, exist_ok=True)
        (ws / "env" / "smoke_report.json").write_text(json.dumps(
            {"python": "3.8.12", "tools": [{"name": t, "status": "ok", "command": f"smoke_test.sh {t}", "output": "passed"} for t in tools]}))
    elif node == "N3":
        wcsv(ws / "data" / "pairs.csv", ["pdb_id", "chain_a", "chain_b", "split"], [list(p) for p in PAIRS])
        wcsv(ws / "data" / "substitutions.csv", ["requested_id", "used_id", "reason"], [])
        n_res_pairs = n_contacts = 0
        for pdb, a, b, split in PAIRS:
            for ch in (a, b):
                xyz = chain_coords(ch)
                wcsv(ws / "data" / "chains" / f"{pdb}_{ch}.csv", ["resnum", "resname", "x", "y", "z"],
                     [[i + 1, ("TYR", "ARG", "ALA")[(i + 1) % 3], *xyz[i]] for i in range(N_RES)])
            D = np.sqrt(((chain_coords(a)[:, None] - chain_coords(b)[None]) ** 2).sum(-1))
            (ws / "data" / "distances").mkdir(parents=True, exist_ok=True)
            np.savez(ws / "data" / "distances" / f"{pdb}_{a}{b}.npz", D=D,
                     res_a=np.array([str(i) for i in range(1, N_RES + 1)]), res_b=np.array([str(i) for i in range(1, N_RES + 1)]))
            if split == "train":
                n_res_pairs += D.size
                n_contacts += int((D < 7).sum())
        (ws / "data" / "dataset_summary.json").write_text(json.dumps({"n_pairs": 2, "n_residue_pairs": n_res_pairs, "n_contacts": n_contacts}))
    elif node == "N3c":
        wcsv(ws / "data" / "np" / "cases.csv", ["case", "np_name", "pdb_id", "chain", "reference"],
             [[c, n, p, ch, "paper Fig. 5"] for c, n, p, ch in CASES])
        for name in {c[1] for c in CASES}:
            wcsv(ws / "data" / "np" / f"{name}_cg.csv", ["node", "x", "y", "z", "n_atoms", "element"],
                 [[i, i * 3.0, 0, 0, 13, "C"] for i in range(6)])
    elif node in ("N4a", "N4b", "N4c"):
        fam = {"N4a": "CH", "N4b": "GE", "N4c": "GT"}[node]
        cols = {"CH": ["c_charge", "hydrophobicity", "polarity", "mw"],
                "GE": ["rinacc", "shell", "pocketness", "opd_5", "opd_7", "opd_10", "opd_15"],
                "GT": ["orc", "frc", "gnm", "mfd", "mfd_1", "mfd_2", "mfd_3", "mfd_4"]}[fam]
        for pdb, a, b, _ in PAIRS:
            for ch in (a, b):
                base = rng.random(N_RES)
                rows = [[i + 1, ("TYR", "ARG", "ALA")[(i + 1) % 3], *[round(base[i] + 0.1 * rng.random(), 4) for _ in cols]] for i in range(N_RES)]
                wcsv(ws / "features" / fam / f"{pdb}_{ch}.csv", ["resnum", "resname", *cols], rows)
    elif node == "N4d":
        cols = ["rinacc", "shell", "pocketness", "opd_5", "opd_7", "orc", "frc", "gnm", "mfd"]
        for name in {c[1] for c in CASES}:
            wcsv(ws / "features" / "np" / f"{name}.csv", ["node", *cols], [[i, *rng.random(len(cols)).round(3)] for i in range(6)])
    elif node == "N5":
        d = ws / "ml" / "datasets"
        d.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for s in SUBSETS:
            fams = {"all": ["CH", "GE", "GT"], "GE_GT": ["GE", "GT"], "CH_GE": ["CH", "GE"], "CH_GT": ["CH", "GT"]}.get(s, [s])
            w = 2 * sum(WIDTH[f] for f in fams)
            n = 200
            np.savez(d / f"{s}.npz", X=rng.random((n, w)), y=rng.integers(0, 2, n), pair_index=np.array([f"1ma9_AB:{i}:{i}" for i in range(n)]))
            manifest[s] = {"n_rows": n, "n_features": w, "feature_names": [f"f{i}" for i in range(w)], "balancing": "undersample 1:1", "seed": 0}
        (d / "manifest.json").write_text(json.dumps(manifest))
    elif node in ("N6a", "N6b"):
        models = ["lr", "gnb", "svm", "rf", "xgb"] if node == "N6a" else ["dnn"]
        d = ws / "ml" / "cv" / "predictions"
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for m in models:
            for s in SUBSETS:
                n = 400
                y = rng.integers(0, 2, n)
                score = np.clip(0.5 + 0.3 * (y - 0.5) + 0.25 * rng.normal(size=n), 0, 1)
                np.savez(d / f"{m}_{s}.npz", fold=np.arange(n) % 10, y_true=y, y_score=score)
                rows += [[m, s, k, 0.7, 0.75, 0.7] for k in range(10)]
        wcsv(ws / "ml" / "cv" / f"metrics_{node}.csv", ["model", "subset", "fold", "accuracy", "auc", "f1"], rows)
        wcsv(ws / "ml" / "cv" / "metrics.csv", ["model", "subset", "fold", "accuracy", "auc", "f1"], rows)
        if node == "N6b":
            (ws / "ml" / "models" / "dnn_GE_GT").mkdir(parents=True, exist_ok=True)
            (ws / "ml" / "models" / "dnn_GE_GT" / "fold0.h5").write_text("fake model")
    elif node == "N7":
        for pdb, a, b, split in PAIRS:
            if split != "heldout":
                continue
            rows = [[i, j, (0.9 if i == j else 0.05) - 1e-6 * (i * N_RES + j)] for i in range(1, N_RES + 1) for j in range(1, N_RES + 1)]
            wcsv(ws / "predictions" / "ppi" / f"{pdb}_dnn_GE_GT.csv", ["res_a", "res_b", "prob"], rows)
            top = sorted(rows, key=lambda r: r[2])[-max(int(len(rows) * 0.001), 1):]
            (ws / "predictions" / "ppi" / f"{pdb}_dnn_GE_GT_interface.json").write_text(
                json.dumps({"chain_a": sorted({r[0] for r in top}), "chain_b": sorted({r[1] for r in top}), "top_fraction": 0.001}))
    elif node == "N8":
        p = ws / "predictions" / "pni"
        p.mkdir(parents=True, exist_ok=True)
        for c, *_ in CASES:
            if "corona" not in c:
                (p / f"{c}_interface.json").write_text(json.dumps({"protein_residues": ["1", "2", "4"], "np_nodes": [0, 1], "top_fraction": 0.001}))
        prots = ["albumin", "apolipoprotein", "histidine-rich glycoprotein", "galectin-3-binding protein"]
        wcsv(p / "corona_ranking.csv", ["np_name", "protein", "score"],
             [["SWNT", pr, 0.2 + 0.1 * k] for k, pr in enumerate(prots)] + [["spherical_C", pr, 0.1 + 0.1 * k] for k, pr in enumerate(prots)])
    elif node == "N9":
        (ws / "report").mkdir(parents=True, exist_ok=True)
        (ws / "report" / "report.md").write_text("# Reproduction report\n\n" + "This synthetic report has words. " * 60)
        claims = [{"claim": f"claim {i}", "verdict": "inconclusive", "evidence": ["ml/cv/metrics.csv"], "notes": "synthetic"} for i in range(5)]
        (ws / "report" / "claims.yaml").write_text(yaml.safe_dump(claims))
    else:
        raise SystemExit(f"no synthetic outputs for {node}")


if __name__ == "__main__":
    write(sys.argv[1], Path(sys.argv[2]))
    print(f"wrote synthetic outputs for {sys.argv[1]}")
