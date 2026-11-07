"""N7 (L3): held-out PPI predictions from the authors' precomputed test features (capsule processed_test_data/PPI)
with the models trained in N6 on the authors' training arrays. The authors' saved models are not used.

Column selection and NaN handling follow PPI_ML_prediction.py ("all" = first 64 columns; forward-fill NaNs).
DNN = mean contact probability of the ten fold models, with the training MinMax scaler applied (the released
script feeds unscaled features to models trained on scaled ones). RF and XGB are refit on the full training arrays.
Rows are ordered chain-A residue major, chain-B minor, as in the released pair tables.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from n6a_classical_cv import make  # noqa: E402

CAP = Path("materials/precomputed")
TOP = 0.001
C1 = ['c1', 'c1_pos', 'c1_neg', 'c1_polar', 'c1_amp', 'c1_hp', 'c1_hp_idx', 'c1_rd', 'c1_shell', 'c1_poc', 'c1_N_count',
      'c1_C_count', 'c1_O_count', 'c1_H_count', 'c1_S_count', 'c1_N_charge', 'c1_C_charge', 'c1_O_charge', 'c1_H_charge',
      'c1_S_charge', 'c1_ollivier', 'c1_forman', 'c1_gnm', 'fd1', 'more_fd1_1', 'more_fd1_2', 'more_fd1_3', 'more_fd1_4',
      'G_os1_5', 'G_os1_7', 'G_os1_10', 'G_os1_15']


def two(c):
    return c.replace('c1', 'c2').replace('fd1', 'fd2').replace('G_os1', 'G_os2')


COLS = C1 + [two(c) for c in C1] + ['distance']
GE_GT1 = ['c1_rd', 'c1_shell', 'c1_poc', 'G_os1_5', 'G_os1_7', 'G_os1_10', 'G_os1_15', 'c1_ollivier', 'c1_forman', 'c1_gnm',
          'fd1', 'more_fd1_1', 'more_fd1_2', 'more_fd1_3', 'more_fd1_4']
GT1 = ['c1_ollivier', 'c1_forman', 'c1_gnm', 'fd1', 'more_fd1_1', 'more_fd1_2', 'more_fd1_3', 'more_fd1_4']
SELECT = {"all": list(range(64)), "GE_GT": [COLS.index(c) for c in GE_GT1 + [two(c) for c in GE_GT1]],
          "GT": [COLS.index(c) for c in GT1 + [two(c) for c in GT1]]}


def features(T, subset):
    return pd.DataFrame(T[:, SELECT[subset]]).fillna(method="ffill").fillna(method="bfill").to_numpy(float)


def dnn(subset, X):
    from tensorflow import keras

    d = Path(f"ml/models/dnn_{subset}")
    sc = np.load(d / "scaler.npz")
    rng = np.where(sc["data_max"] > sc["data_min"], sc["data_max"] - sc["data_min"], 1.0)
    Xs = (X - sc["data_min"]) / rng
    return np.mean([keras.models.load_model(str(f), compile=False).predict(Xs, batch_size=8192)[:, 1]
                    for f in sorted(d.glob("fold*.h5"))], axis=0)


def write(out, pdb, model, subset, s, ra, rb):
    ia, ib = np.divmod(np.arange(len(s)), len(rb))
    with (out / f"{pdb}_{model}_{subset}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["res_a", "res_b", "prob"])
        for i, j, p in zip(ia, ib, s):
            w.writerow([ra[i], rb[j], f"{p:.6f}"])
    n = max(int(len(s) * TOP), 1)
    top = np.argsort(s)[-n:]
    iface = {"chain_a": sorted({ra[ia[k]] for k in top}, key=lambda x: (len(x), x)),
             "chain_b": sorted({rb[ib[k]] for k in top}, key=lambda x: (len(x), x)), "top_fraction": TOP, "n_pairs_scored": int(len(s))}
    (out / f"{pdb}_{model}_{subset}_interface.json").write_text(json.dumps(iface))
    return iface


def main():
    out = Path("predictions/ppi")
    out.mkdir(parents=True, exist_ok=True)
    for subset in ["GE_GT", "GT", "all"]:
        z = np.load(f"ml/datasets/{subset}.npz")
        fitted = {m: make(m).fit(z["X"], z["y"]) for m in ("rf", "xgb")}
        for pdb in ["6wzo", "6nqn", "1c29"]:
            T = np.load(CAP / f"processed_test_data/PPI/data_feature_{pdb}_AB.npy", allow_pickle=True).astype(float)
            ra = [r["resnum"] for r in csv.DictReader(open(f"data/chains/{pdb}_A.csv"))]
            rb = [r["resnum"] for r in csv.DictReader(open(f"data/chains/{pdb}_B.csv"))]
            assert len(T) == len(ra) * len(rb), (pdb, len(T), len(ra), len(rb))
            X = features(T, subset)
            scores = {"dnn": dnn(subset, X), **{m: f.predict_proba(X)[:, 1] for m, f in fitted.items()}}
            for m, s in scores.items():
                iface = write(out, pdb, m, subset, s, ra, rb)
                print(pdb, m, subset, "interface", len(iface["chain_a"]), len(iface["chain_b"]), flush=True)


if __name__ == "__main__":
    main()
