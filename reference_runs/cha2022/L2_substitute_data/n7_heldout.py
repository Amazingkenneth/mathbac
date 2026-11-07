"""N7: interface predictions for the held-out complexes 6WZO, 6NQN and 1C29.

DNN: mean probability of the ten fold models from N6b (as the released PPI_ML_prediction.py ensembles its ten
saved models), applied to features scaled with the training scaler. RF and XGB: refit on the full training set.
Every residue pair of each complex is scored; the top 0.1% of pairs by probability give the interface residues.

    /opt/mathbac/env/bin/python scripts/n7_heldout.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from n5_datasets import SUBSETS, pair_features  # noqa: E402
from n6a_classical_cv import make  # noqa: E402

TOP = 0.001
SUBS = ["GE_GT", "GT", "all"]


def dnn_scores(subset, X):
    from tensorflow import keras

    d = Path(f"ml/models/dnn_{subset}")
    sc = np.load(d / "scaler.npz")
    rng = np.where(sc["data_max"] > sc["data_min"], sc["data_max"] - sc["data_min"], 1.0)
    Xs = (X - sc["data_min"]) / rng
    q = [keras.models.load_model(str(f), compile=False).predict(Xs, batch_size=4096)[:, 0].astype(np.float64) for f in sorted(d.glob("fold*.h5"))]
    return 1.0 - np.mean(q, axis=0)  # float64 from P(class 0): P(class 1) saturates at 1.0 in float32 and ties the top pairs


def main():
    held = [r for r in csv.DictReader(open("data/pairs.csv")) if r["split"] == "heldout"]
    out = Path("predictions/ppi")
    out.mkdir(parents=True, exist_ok=True)
    for subset in SUBS:
        z = np.load(f"ml/datasets/{subset}.npz")
        fitted = {m: make(m).fit(z["X"], z["y"]) for m in ("rf", "xgb")}
        for r in held:
            pdb, a, b = r["pdb_id"].lower(), r["chain_a"], r["chain_b"]
            X, ia, ib = pair_features(pdb, a, b, SUBSETS[subset])
            ra = [x["resnum"] for x in csv.DictReader(open(f"data/chains/{pdb}_{a}.csv"))]
            rb = [x["resnum"] for x in csv.DictReader(open(f"data/chains/{pdb}_{b}.csv"))]
            scores = {"dnn": dnn_scores(subset, X), **{m: f.predict_proba(X)[:, 1] for m, f in fitted.items()}}
            for m, s in scores.items():
                with (out / f"{pdb}_{m}_{subset}.csv").open("w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["res_a", "res_b", "prob"])
                    for i, j, p in zip(ia, ib, s):
                        w.writerow([ra[i], rb[j], repr(float(p))])
                n = max(int(len(s) * TOP), 1)
                top = np.where(s >= np.sort(s)[-n])[0]  # every pair tied with the last one included
                iface = {"chain_a": sorted({ra[ia[k]] for k in top}, key=lambda x: (len(x), x)),
                         "chain_b": sorted({rb[ib[k]] for k in top}, key=lambda x: (len(x), x)), "top_fraction": TOP,
                         "n_pairs_scored": int(len(s))}
                (out / f"{pdb}_{m}_{subset}_interface.json").write_text(json.dumps(iface))
                print(pdb, m, subset, "interface residues", len(iface["chain_a"]), len(iface["chain_b"]), flush=True)


if __name__ == "__main__":
    main()
