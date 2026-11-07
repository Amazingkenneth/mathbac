"""N6a: tenfold CV of LR, GNB, SVM, RF and XGB on every descriptor subset.

KFold(10, shuffle=True, random_state=0) as in the released DNN_train.py. RF uses the released settings
(200 trees, entropy, max depth 50). Features are MinMax-scaled inside each training fold. SVM is an RBF SVC
with Platt probabilities. Writes ml/cv/predictions/<model>_<subset>.npz and ml/cv/metrics_classical.csv.

    /opt/mathbac/env/bin/python scripts/n6a_classical_cv.py [--models lr,gnb,svm,rf,xgb]
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

SUBSETS = ["all", "CH", "GE", "GT", "GE_GT", "CH_GE", "CH_GT"]


SCALE = True


def make(name):
    pre = [MinMaxScaler()] if SCALE else []
    if name == "lr":
        return make_pipeline(*pre, LogisticRegression(max_iter=2000))
    if name == "gnb":
        return make_pipeline(*pre, GaussianNB())
    if name == "svm":
        return make_pipeline(*pre, SVC(probability=True, random_state=0))
    if name == "svmd":  # SVM without Platt calibration: scores = logistic of the decision function (same ranking, ~6x faster)
        return make_pipeline(*pre, SVC(probability=False, random_state=0))
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, criterion="entropy", max_depth=50, n_jobs=4, random_state=0)
    if name == "xgb":
        return XGBClassifier(n_jobs=4, random_state=0, use_label_encoder=False, eval_metric="logloss")
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lr,gnb,svm,rf,xgb")
    ap.add_argument("--scale", choices=["fold", "none"], default="fold",
                    help="none: no feature scaling for LR/GNB/SVM (the paper does not say whether it scaled)")
    ap.add_argument("--out", default="ml/cv/predictions")
    a = ap.parse_args()
    global SCALE
    SCALE = a.scale == "fold"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for subset in SUBSETS:
        z = np.load(f"ml/datasets/{subset}.npz")
        X, y = z["X"], z["y"]
        for m in a.models.split(","):
            t0 = time.time()
            folds, ys, ss = [], [], []
            for k, (tr, te) in enumerate(KFold(10, shuffle=True, random_state=0).split(X)):
                clf = make(m).fit(X[tr], y[tr])
                s = clf.predict_proba(X[te])[:, 1] if m != "svmd" else 1 / (1 + np.exp(-clf.decision_function(X[te])))
                folds.append(np.full(len(te), k)); ys.append(y[te]); ss.append(s)
                rows.append([m, subset, k, accuracy_score(y[te], s >= 0.5), roc_auc_score(y[te], s), f1_score(y[te], s >= 0.5)])
            np.savez(out / f"{'svm' if m == 'svmd' else m}_{subset}.npz", fold=np.concatenate(folds), y_true=np.concatenate(ys), y_score=np.concatenate(ss))
            r = [x for x in rows if x[0] == m and x[1] == subset]
            print(f"{m:4s} {subset:6s} acc {np.mean([x[3] for x in r]):.3f} auc {np.mean([x[4] for x in r]):.3f} "
                  f"f1 {np.mean([x[5] for x in r]):.3f}  ({time.time() - t0:.0f}s)", flush=True)
    with open(Path(a.out).parent / f"metrics_classical_{a.scale}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "subset", "fold", "accuracy", "auc", "f1"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
