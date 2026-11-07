"""N6: ml/cv/metrics.csv (per-fold accuracy, AUC and non-contact-class F1) from ml/cv/predictions/*.npz."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from n9_report import auc  # noqa: E402

with open("ml/cv/metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "subset", "fold", "accuracy", "auc", "f1"])
    for p in sorted(Path("ml/cv/predictions").glob("*.npz")):
        m, s = p.stem.split("_", 1)
        z = np.load(p)
        for k in np.unique(z["fold"]):
            i = z["fold"] == k; y = z["y_true"][i].astype(int); sc = z["y_score"][i]
            pn, yn = sc < 0.5, y == 0
            tp = (pn & yn).sum(); fp = (pn & ~yn).sum(); fn = (~pn & yn).sum()
            w.writerow([m, s, int(k), ((sc >= 0.5) == (y == 1)).mean(), auc(y, sc), 2 * tp / (2 * tp + fp + fn)])
