"""N5: one dataset per descriptor subset with consistent shapes and binary labels."""

from pathlib import Path

import numpy as np

from mathbac.contracts import load_json, load_npz

SUBSETS = ["all", "CH", "GE", "GT", "GE_GT", "CH_GE", "CH_GT"]


def check(workspace: Path, node) -> list[str]:
    d = workspace / "ml" / "datasets"
    try:
        manifest = load_json(d / "manifest.json")
    except Exception as e:
        return [f"ml/datasets/manifest.json unreadable: {e}"]
    problems = []
    widths = {}
    for s in SUBSETS:
        if s not in manifest:
            problems.append(f"manifest.json: no entry for subset {s}")
        f = d / f"{s}.npz"
        if not f.exists():
            problems.append(f"missing ml/datasets/{s}.npz")
            continue
        arr, errs = load_npz(f, ["X", "y", "pair_index"])
        if errs:
            problems += errs
            continue
        X, y = arr["X"], arr["y"]
        if X.ndim != 2 or len(X) != len(y) or len(y) != len(arr["pair_index"]):
            problems.append(f"{s}.npz: X, y and pair_index must have the same number of rows")
            continue
        if not set(np.unique(y)) <= {0, 1}:
            problems.append(f"{s}.npz: y must be 0/1")
        widths[s] = X.shape[1]
    if {"GE", "GT", "GE_GT"} <= widths.keys() and widths["GE_GT"] != widths["GE"] + widths["GT"]:
        problems.append(f"GE_GT has {widths['GE_GT']} features but GE + GT have {widths['GE'] + widths['GT']}")
    return problems
