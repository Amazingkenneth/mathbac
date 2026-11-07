"""N6a/N6b: raw per-fold predictions for every (model, subset) the node was asked for."""

from pathlib import Path

import numpy as np

from mathbac.contracts import load_npz

SUBSETS = ["all", "CH", "GE", "GT", "GE_GT", "CH_GE", "CH_GT"]


def check(workspace: Path, node) -> list[str]:
    models = node.facts.get("models") or []
    folds = int(node.facts.get("cv_folds", 10))
    problems = []
    for m in models:
        for s in SUBSETS:
            f = workspace / "ml" / "cv" / "predictions" / f"{m}_{s}.npz"
            if not f.exists():
                problems.append(f"missing ml/cv/predictions/{f.name}")
                continue
            arr, errs = load_npz(f, ["fold", "y_true", "y_score"])
            if errs:
                problems += errs
                continue
            if not (len(arr["fold"]) == len(arr["y_true"]) == len(arr["y_score"])):
                problems.append(f"{f.name}: fold, y_true and y_score lengths differ")
                continue
            n = len(np.unique(arr["fold"]))
            if n != folds:
                problems.append(f"{f.name}: {n} folds, expected {folds}")
            ys = arr["y_score"]
            if ys.size and (np.nanmin(ys) < 0 or np.nanmax(ys) > 1):
                problems.append(f"{f.name}: y_score must be a probability in [0, 1]")
    return problems
