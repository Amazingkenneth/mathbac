"""Classification metrics in plain numpy, so the evaluator does not depend on the agents' ML stack."""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true).astype(int)
    return float(np.mean((np.asarray(y_score) >= threshold).astype(int) == y_true))


def f1(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney U formulation with average ranks for ties."""
    y_true = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def per_fold(fold: np.ndarray, y_true: np.ndarray, y_score: np.ndarray, f1_class: str = "contact") -> dict[str, tuple[float, float]]:
    """Mean and s.d. of accuracy, AUC and F1 across folds.

    f1_class="non_contact" scores F1 for the non-contact class (y = 0), which is what the paper's Table 1 reports:
    the released code calls sklearn's f1_score with its default positive label 1, and label 1 is non-contact there.
    """
    vals: dict[str, list[float]] = {"accuracy": [], "auc": [], "f1": []}
    for k in np.unique(fold):
        m = fold == k
        vals["accuracy"].append(accuracy(y_true[m], y_score[m]))
        vals["auc"].append(roc_auc(y_true[m], y_score[m]))
        if f1_class == "non_contact":
            yt = 1 - np.asarray(y_true[m]).astype(int)
            pred_non = (np.asarray(y_score[m]) < 0.5).astype(float)
            vals["f1"].append(f1(yt, pred_non))
        else:
            vals["f1"].append(f1(y_true[m], y_score[m]))
    return {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in vals.items()}


def set_scores(pred: set, true: set, universe: set) -> dict[str, float]:
    """Per-element classification of a predicted set against a true set within a universe."""
    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)
    tn = len(universe - pred - true)
    n = max(len(universe), 1)
    return {
        "accuracy": (tp + tn) / n,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
