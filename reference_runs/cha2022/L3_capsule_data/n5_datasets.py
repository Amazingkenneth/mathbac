"""N5: residue-pair feature matrices for the seven descriptor subsets.

Following the released code: a pair's features are chain A's residue descriptors followed by chain B's;
label 1 = contact (Cα-Cα distance <= 7 Å, distance class 0 in the released code). Non-contacts are
undersampled to a 1:1 ratio (the released training arrays are named binary_under_*), seed 0.

Per-residue widths match the released arrays: CH 17 (5 class flags, Kyte-Doolittle, MW, 5 element charges,
5 element counts), GE 7 (rinacc, shell, pocketness, OPD 5/7/10/15), GT 8 (ORC, FRC, GNM, MFD + 4) -> pair widths
all 64, CH 34, GE 14, GT 16, GE_GT 30, CH_GE 48, CH_GT 50, as in DNN_train.py.
"""

import csv
import json
from pathlib import Path

import numpy as np

COLS = {
    "CH": ["pos", "neg", "polarity", "amp", "hp", "hydrophobicity", "mw", "n_charge", "c_charge", "o_charge", "h_charge",
           "s_charge", "n_count", "c_count", "o_count", "h_count", "s_count"],
    "GE": ["rinacc", "shell", "pocketness", "opd_5", "opd_7", "opd_10", "opd_15"],
    "GT": ["orc", "frc", "gnm", "mfd", "mfd_1", "mfd_2", "mfd_3", "mfd_4"],
}
SUBSETS = {"all": ["CH", "GE", "GT"], "CH": ["CH"], "GE": ["GE"], "GT": ["GT"], "GE_GT": ["GE", "GT"],
           "CH_GE": ["CH", "GE"], "CH_GT": ["CH", "GT"]}


def residue_matrix(pdb, ch, fams):
    blocks = []
    for fam in fams:
        rows = list(csv.DictReader(open(f"features/{fam}/{pdb}_{ch}.csv")))
        blocks.append(np.array([[float(r[c]) for c in COLS[fam]] for r in rows], float))
    return np.hstack(blocks)


def pair_features(pdb, a, b, fams, ia=None, ib=None):
    A, B = residue_matrix(pdb, a, fams), residue_matrix(pdb, b, fams)
    if ia is None:
        ia, ib = np.meshgrid(np.arange(len(A)), np.arange(len(B)), indexing="ij")
        ia, ib = ia.ravel(), ib.ravel()
    return np.hstack([A[ia], B[ib]]), ia, ib


def main():
    rng = np.random.default_rng(0)
    pairs = [r for r in csv.DictReader(open("data/pairs.csv")) if r["split"] == "train"]
    # choose the sampled rows once, so every subset has exactly the same pairs
    picks = []
    for r in pairs:
        pdb, a, b = r["pdb_id"].lower(), r["chain_a"], r["chain_b"]
        D = np.load(f"data/distances/{pdb}_{a}{b}.npz")["D"]
        pos = np.argwhere(D <= 7.0)
        neg = np.argwhere(D > 7.0)
        take = rng.choice(len(neg), size=min(len(pos), len(neg)), replace=False)
        for (i, j), y in [((i, j), 1) for i, j in pos] + [((i, j), 0) for i, j in neg[take]]:
            picks.append((pdb, a, b, int(i), int(j), y))
    out = Path("ml/datasets")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, fams in SUBSETS.items():
        X, Y, idx = [], [], []
        by_complex = {}
        for p in picks:
            by_complex.setdefault(p[:3], []).append(p)
        for (pdb, a, b), rows in by_complex.items():
            ia = np.array([r[3] for r in rows]); ib = np.array([r[4] for r in rows])
            F, _, _ = pair_features(pdb, a, b, fams, ia, ib)
            X.append(F); Y += [r[5] for r in rows]; idx += [f"{pdb}_{a}{b}:{r[3]}:{r[4]}" for r in rows]
        X = np.vstack(X)
        names = [f"{side}_{c}" for side in ("a", "b") for fam in fams for c in COLS[fam]]
        assert not np.isnan(X).any(), f"NaN features in subset {name}"
        np.savez(out / f"{name}.npz", X=X, y=np.array(Y, int), pair_index=np.array(idx))
        manifest[name] = {"n_rows": int(len(Y)), "n_features": int(X.shape[1]), "feature_names": names,
                          "balancing": "all contacts (<= 7 Å) plus an equal number of randomly drawn non-contacts per complex",
                          "seed": 0, "n_complexes": len(by_complex), "n_positive": int(sum(Y))}
        print(name, X.shape)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
