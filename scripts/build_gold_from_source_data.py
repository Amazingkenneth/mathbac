"""Build gold files for the cha2022 checkpoints from the paper's published source data.

    python scripts/build_gold_from_source_data.py <source_data_dir> [--pdb-cache cache/pdb]

<source_data_dir> holds the unzipped Springer ESM archives (Fig1_Source_Data ... Fig5_Source_Data)
and 43588_2022_229_MOESM2_ESM.xlsx (per-fold scores). Download them from
https://static-content.springer.com/esm/art%3A10.1038%2Fs43588-022-00229-w/MediaObjects/43588_2022_229_MOESM<k>_ESM.<ext>
(k=2 xlsx; k=3..7 zip).

Writes into mathbac/eval/gold/cha2022/:
  reference_descriptors/1ma9_A.csv   Fig. 1d-n per-residue values, keyed by PDB residue number
  fig2a_correlations.yaml            Fig. 2a descriptor cross-correlations (whole proteins)
  table1_folds.csv                   per-fold accuracy / AUC / F1 for every model and subset
  heldout_authors.yaml               Fig. 4 ground-truth and predicted interface residues
  pni_authors.yaml                   Fig. 5 predicted residues (mapped to residue numbers) and Fig. 5f probabilities

Mapping decisions (checked against RCSB coordinates):
- Fig. 1 rows are 1MA9 chain A's 442 residues, numbered by PDB residue number; Fig. 1b's matrix matches RCSB to 1e-14 Å.
- Fig. 1 "OPD" is opd_7 and Fig. 1 "MFD" is Forman-Ricci curvature (exact matches with the released code).
- Fig. 5d/e hMPO indices are MPO residue numbers over chain A (1-104) and chain C (113-578).
- Fig. 5a PSM-α indices are atom positions in the authors' 5I55 file; mapped to residue numbers.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mathbac.science import pdb  # noqa: E402

GOLD = ROOT / "mathbac" / "eval" / "gold" / "cha2022"


def sheet_rows(path: Path, sheet: str | None = None) -> list[tuple]:
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet] if sheet else wb.worksheets[0]
    return [r for r in ws.iter_rows(values_only=True) if any(c is not None for c in r)]


def columns(rows: list[tuple]) -> list[list]:
    return [[x for x in col[1:] if x is not None] for col in zip(*rows)]


def build_descriptors(src: Path, cache: Path) -> None:
    rows = sheet_rows(src / "Fig1_Source_Data" / "Figure1_defhijlmn.xlsx")
    header, body = rows[0], rows[1:]
    chain = pdb.load_chain("1MA9", "A", cache)
    assert len(body) == len(chain.resnums), (len(body), len(chain.resnums))
    # Verified with the released code on 1MA9 chain A (exact matches): "OPD" is the n=7 index, and the column
    # labelled "MFD" (Fig. 1n, "multifractal dimension") is Forman-Ricci node curvature, so it is stored as frc.
    rename = {"C_charge": "c_charge", "Hp": "hydrophobicity", "MW": "mw", "Rinacc": "rinacc", "Pocket": "pocketness",
              "OPD": "opd_7", "ORC": "orc", "GNM": "gnm", "MFD": "frc"}
    out = GOLD / "reference_descriptors"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "1ma9_A.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["resnum", "resname", *[rename[h] for h in header[1:]]])
        for i, r in enumerate(body):
            assert str(r[0]) == chain.resnums[i], (r[0], chain.resnums[i])  # Fig. 1 uses PDB residue numbers
            w.writerow([chain.resnums[i], chain.resnames[i], *r[1:]])
    print(f"reference descriptors: {len(body)} residues of 1MA9 chain A")


def build_capsule_reference(example: Path, cache: Path) -> None:
    """Optional: the capsule's data/1ma9_example (the authors' own per-residue arrays for both 1MA9 chains).

    Adds mfd (absent from Fig. 1) and chain B. Verified: for chain A these arrays equal the Fig. 1 values, and the
    released code reproduces orc, frc, gnm, opd_7 and mfd exactly.
    """
    import numpy as np

    arrays = {"c_charge": "c_charge_1ma9_{c}.npy", "hydrophobicity": "hp_1ma9_{c}.npy", "orc": "ollivier_1ma9_{c}.npy",
              "frc": "forman_1ma9_{c}.npy", "gnm": "gnm_1ma9_{c}.npy", "opd_7": "opd_1ma9_{c}_7.npy", "mfd": "fd_1ma9_{c}.npy"}
    out = GOLD / "reference_descriptors"
    out.mkdir(parents=True, exist_ok=True)
    for c in ("A", "B"):
        chain = pdb.load_chain("1MA9", c, cache)
        cols = {k: np.asarray(np.load(example / f.format(c=c), allow_pickle=True), float) for k, f in arrays.items()}
        ge = {}
        for line in open(example / f"ge_1ma9_{c}.txt"):
            t = line.split()
            if t and not line.startswith("#"):
                ge.setdefault(t[0], (float(t[4]), float(t[3]), float(t[7])))
        n = len(chain.resnums)
        assert all(len(v) == n for v in cols.values()), {k: len(v) for k, v in cols.items()}
        with (out / f"1ma9_{c}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["resnum", "resname", *cols, "rinacc", "shell", "pocketness"])
            for i, rn in enumerate(chain.resnums):
                g = ge.get(rn, ("", "", ""))
                w.writerow([rn, chain.resnames[i], *[cols[k][i] for k in cols], *g])
    print("reference descriptors: 1MA9 chains A and B from the capsule example (adds mfd and shell)")


def build_fig2a(src: Path) -> None:
    rows = sheet_rows(src / "Fig2_Source_Data" / "Figure2_ab.xlsx")
    names = []
    for n in rows[1][1:]:  # contiguous header cells; anything after the first gap belongs to another table
        if n is None:
            break
        names.append(n)
    matrix = {}
    for r in rows[2 : 2 + len(names)]:
        matrix[r[0]] = {n: round(float(v), 4) for n, v in zip(names, r[1 : 1 + len(names)])}
    (GOLD / "fig2a_correlations.yaml").write_text(yaml.safe_dump({"source": "Fig. 2a, whole proteins", "matrix": matrix}, sort_keys=False))
    print(f"Fig. 2a correlations: {len(names)} descriptors")


SUBSET = {"All": "all", "All ": "all", "CH": "CH", "GE": "GE", "GT": "GT", "GE+GT": "GE_GT", "CH+GE": "CH_GE", "CH+GT": "CH_GT",
          "GE_GT": "GE_GT", "CH_GE": "CH_GE", "CH_GT": "CH_GT", "GE+CH": "CH_GE", "GT+CH": "CH_GT"}
MODEL = {"Logistic_Regression": "lr", "Gaussian_Naive_Bayes": "gnb", "Support_Vector_Machine": "svm", "Random_Forest": "rf", "XGBoost": "xgb"}
METRIC = {"Accuracy": "accuracy", "AUC": "auc", "F1": "f1", "F1 score": "f1", "F1 Score": "f1", "F1_score": "f1"}


def build_folds(src: Path) -> None:
    out = []
    xlsx = src / "43588_2022_229_MOESM2_ESM.xlsx"
    import openpyxl

    for model, path, sheets in [(None, xlsx, None), ("dnn", src / "Fig3_Source_Data" / "Figure3_bc.xlsx", ["Fig3c"])]:
        wb = openpyxl.load_workbook(path, data_only=True)
        for ws in wb.worksheets:
            if sheets and ws.title not in sheets:
                continue
            m = model or MODEL.get(ws.title)
            if not m:
                continue
            counter: dict[tuple, int] = {}
            for r in list(ws.iter_rows(values_only=True))[1:]:
                if r[0] is None or r[1] is None:
                    continue
                s, metric = SUBSET.get(str(r[0])), METRIC.get(str(r[2]).strip())
                if not s or not metric:
                    continue
                k = (s, metric)
                counter[k] = counter.get(k, 0) + 1
                out.append([m, s, metric, counter[k] - 1, float(r[1])])
    # Fig. 3c labels the DNN "GT" and "GE+CH" rows the other way round from Table 1 and the main text
    # (text: GT-only DNN AUC 87.7%; CH adds 1.9 points to GE). Table 1 is used as gold; rows are kept as published.
    with (GOLD / "table1_folds.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "subset", "metric", "fold", "value"])
        w.writerows(out)
    print(f"per-fold scores: {len(out)} rows")


def build_heldout(src: Path) -> None:
    out = {}
    for fname, pid in [("Figure4_ab.xlsx", "6WZO"), ("Figure4_cd.xlsx", "6NQN"), ("Figure4_ef.xlsx", "1C29")]:
        import openpyxl

        wb = openpyxl.load_workbook(src / "Fig4_Source_Data" / fname, data_only=True)
        truth, pred = [columns([r for r in ws.iter_rows(values_only=True) if any(c is not None for c in r)]) for ws in wb.worksheets]
        out[pid] = {"truth": {"A": [str(x) for x in truth[0]], "B": [str(x) for x in truth[1]]},
                    "dnn_GE_GT_predicted": {"A": [str(x) for x in pred[0]], "B": [str(x) for x in pred[1]]}}
    # SI Tables 4-6 (GE+GT DNN rows), as printed; their metric definition is not reconstructable from the data.
    out["_si_reported"] = {"6WZO": {"accuracy": 0.721, "precision": 0.881, "recall": 0.667, "f1": 0.759},
                           "6NQN": {"accuracy": 0.803, "precision": 0.773, "recall": 0.823, "f1": 0.797},
                           "1C29": {"accuracy": 0.786, "precision": 0.727, "recall": 0.823, "f1": 0.772}}
    (GOLD / "heldout_authors.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    print("held-out interfaces: 3 complexes")


def build_pni(src: Path) -> None:
    d = src / "Fig5_Source_Data"
    out: dict = {}
    hmpo = columns(sheet_rows(d / "Figure5_de.xlsx"))
    out["CNT-1CXP"] = {"pdb": "1CXP", "chains": ["A", "C"], "numbering": "MPO residue numbers (chain A 1-104, chain C 113-578)",
                       "protein_residues": [str(x) for x in hmpo[0]], "np_nodes": hmpo[1]}
    psma = columns(sheet_rows(d / "Figure5_a.xlsx"))
    atoms = [l for l in (d / "PDB" / "Fig5a_5i55_0.pdb").read_text().splitlines() if l.startswith(("ATOM", "HETATM"))]
    res = sorted({int(atoms[i - 1][22:26]) for i in psma[0]})
    out["GQD-PSMa"] = {"pdb": "5I55", "chains": ["A"], "numbering": "atom positions in the authors' 5I55 file, mapped to residues",
                       "protein_residues": [str(x) for x in res], "n_residues": 21,
                       "note": "Paper text names PDB 5KHB and says the GQD docks near the N-terminus; the source data use 5I55 and the predicted residues are 11-21 of 2-22."}
    for key, fname, pid in [("GQD-2L86", "Figure5_b.xlsx", "2L86"), ("GQD-6ZRF", "Figure5_c.xlsx", "6ZRF")]:
        cols = columns(sheet_rows(d / fname))
        out[key] = {"pdb": pid, "numbering": "authors' atomic/combined file numbering (see Fig5_Source_Data/PDB)",
                    "protein_indices": cols[0], "np_nodes": cols[1]}
    corona = sheet_rows(d / "Figure5_f.xlsx")[1:]
    out["corona"] = [{"protein": r[1], "np": r[3], "probability": float(r[2])} for r in corona]
    (GOLD / "pni_authors.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    print("protein-NP predictions:", ", ".join(k for k in out if k != "corona"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source_data", type=Path)
    ap.add_argument("--pdb-cache", type=Path, default=ROOT / "cache" / "pdb")
    ap.add_argument("--capsule-example", type=Path, help="capsule data/1ma9_example folder (optional, more complete reference)")
    a = ap.parse_args()
    build_descriptors(a.source_data, a.pdb_cache)
    if a.capsule_example:
        build_capsule_reference(a.capsule_example, a.pdb_cache)
    build_fig2a(a.source_data)
    build_folds(a.source_data)
    build_heldout(a.source_data)
    build_pni(a.source_data)


if __name__ == "__main__":
    main()
