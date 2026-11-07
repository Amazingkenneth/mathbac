"""Checkpoints C0-C7 for the Cha et al. 2022 reproduction.

Every checker recomputes from the workspace files and gold data the agents never
see. Nothing an agent reports in a message or summary is trusted.

Status values: pass | fail | needs_review | not_scored.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from ..contracts import load_npz, read_csv
from ..science import pdb
from . import metrics


@dataclass
class CheckResult:
    id: str
    title: str
    status: str
    score: float
    details: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalContext:
    run_dir: Path
    workspace: Path
    gold_dir: Path
    pdb_cache: Path
    access_level: str = "L3"
    reviews: dict[str, str] = field(default_factory=dict)

    def gold(self, name: str) -> Any:
        return yaml.safe_load((self.gold_dir / name).read_text())


def _f(x: Any) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _flat_text(v: Any) -> str:
    if isinstance(v, dict):
        return " ".join(f"{k} {_flat_text(x)}" for k, x in v.items()).lower()
    if isinstance(v, (list, tuple)):
        return " ".join(_flat_text(x) for x in v).lower()
    return str(v).lower()


# ---------------------------------------------------------------------------
def c0_environment(ctx: EvalContext) -> CheckResult:
    p = ctx.workspace / "env" / "smoke_report.json"
    if not p.exists():
        return CheckResult("C0", "Environment and tools run", "fail", 0.0, ["env/smoke_report.json missing"])
    rep = json.loads(p.read_text())
    tools = [t for t in rep.get("tools", []) if isinstance(t, dict)]
    ok = [t["name"] for t in tools if t.get("status") == "ok"]
    bad = [f"{t.get('name')}: {t.get('status')}" for t in tools if t.get("status") != "ok"]
    score = len(ok) / len(tools) if tools else 0.0
    details = [f"{len(ok)}/{len(tools)} smoke tests passed"] + bad + \
        ["Self-reported by the agent; rerun the card smoke tests in the image to confirm."]
    return CheckResult("C0", "Environment and tools run", "pass" if tools and not bad else "fail", score, details)


# ---------------------------------------------------------------------------
def _spec_value(spec: dict, dotted: str) -> Any:
    head, *rest = dotted.split(".")
    entry = spec.get(head)
    if entry is None:
        return None
    val = entry.get("value") if isinstance(entry, dict) and "value" in entry else entry
    for k in rest:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return None
    return val


def _match(item: dict, value: Any) -> bool:
    if value is None:
        return False
    kind = item["match"]
    if kind == "number":
        v = _f(value if not isinstance(value, dict) else value.get("paper", value.get("code")))
        return v is not None and abs(v - float(item["expected"])) <= float(item.get("tol", 0))
    text = _flat_text(value)
    if kind == "text":
        return any(s.lower() in text for s in item["any"])
    if kind == "all_of":
        return all(any(s.lower() in text for s in group) for group in item["groups"])
    if kind == "set":
        got = value if isinstance(value, list) else [value]
        return {str(x).lower() for x in got} == {str(x).lower() for x in item["expected"]}
    raise ValueError(f"unknown match type {kind}")


def c1_spec(ctx: EvalContext) -> CheckResult:
    p = ctx.workspace / "spec" / "spec.yaml"
    if not p.exists():
        return CheckResult("C1", "Method spec extracted", "fail", 0.0, ["spec/spec.yaml missing"])
    try:
        spec = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        return CheckResult("C1", "Method spec extracted", "fail", 0.0, [f"spec.yaml invalid: {e}"])
    gold = ctx.gold("spec_checklist.yaml")
    items = [i for i in gold["items"] if not (i.get("source") == "code" and ctx.access_level == "L1")]
    hits, misses = [], []
    for it in items:
        (hits if _match(it, _spec_value(spec, it["key"])) else misses).append(it["id"])
    score = len(hits) / len(items)
    status = "pass" if score >= gold["pass_fraction"] else "fail"
    return CheckResult("C1", "Method spec extracted", status, score,
                       [f"{len(hits)}/{len(items)} checklist items correct (pass at {gold['pass_fraction']:.0%})"] +
                       ([f"missing or wrong: {', '.join(misses)}"] if misses else []),
                       {"hits": hits, "misses": misses})


# ---------------------------------------------------------------------------
def c2_data(ctx: EvalContext) -> CheckResult:
    gold = ctx.gold("heldout.yaml")
    data = ctx.workspace / "data"
    if not (data / "pairs.csv").exists():
        return CheckResult("C2", "Training data rebuilt", "fail", 0.0, ["data/pairs.csv missing"])
    _, pairs = read_csv(data / "pairs.csv")
    train = [r for r in pairs if r.get("split") == "train"]
    checks: dict[str, bool | None] = {}
    details = []

    checks["n_pairs"] = len(train) == gold["dataset"]["n_pairs"]
    details.append(f"training pairs: {len(train)} (paper: {gold['dataset']['n_pairs']})")

    total = 0
    missing = 0
    for r in train:
        f = data / "distances" / f"{r['pdb_id'].lower()}_{r['chain_a']}{r['chain_b']}.npz"
        arr, errs = load_npz(f, ["D"]) if f.exists() else ({}, ["missing"])
        if errs:
            missing += 1
            continue
        total += int(arr["D"].size)
    target = gold["dataset"]["n_residue_pairs"]
    rel = abs(total - target) / target
    checks["n_residue_pairs"] = missing == 0 and rel <= gold["dataset"]["residue_pairs_rel_tol"]
    details.append(f"residue pairs recomputed from distance matrices: {total:,} (paper: {target:,}; off by {rel:.1%})"
                   + (f"; {missing} matrices missing" if missing else ""))

    dc = gold["distance_check"]
    f = data / "distances" / f"{dc['pdb'].lower()}_{dc['chain_a']}{dc['chain_b']}.npz"
    if not f.exists():
        checks["distance_1ma9"] = False
        details.append(f"{f.name} missing")
    else:
        try:
            a = pdb.load_chain(dc["pdb"], dc["chain_a"], ctx.pdb_cache)
            b = pdb.load_chain(dc["pdb"], dc["chain_b"], ctx.pdb_cache)
            ref = pdb.distance_matrix(a.ca, b.ca)
            arr, _ = load_npz(f, ["D"])
            if arr["D"].shape != ref.shape:
                checks["distance_1ma9"] = False
                details.append(f"1MA9 matrix shape {arr['D'].shape}, RCSB gives {ref.shape}")
            else:
                diff = float(np.abs(arr["D"] - ref).max())
                checks["distance_1ma9"] = diff <= dc["max_abs_diff_A"]
                details.append(f"1MA9 distance matrix max deviation from RCSB coordinates: {diff:.4f} Å")
        except (OSError, ValueError) as e:
            checks["distance_1ma9"] = None
            details.append(f"1MA9 check not run: {e}")
    scored = [v for v in checks.values() if v is not None]
    score = sum(scored) / len(scored) if scored else 0.0
    status = "pass" if scored and all(scored) else "fail"
    return CheckResult("C2", "Training data rebuilt", status, score, details, {"checks": checks})


# ---------------------------------------------------------------------------
FAMILIES = {
    "CH": ["charge", "hydrophobicity", "polarity", "mw"],
    "GE": ["rinacc", "shell", "pocketness", "opd_5", "opd_7", "opd_10", "opd_15"],
    "GT": ["orc", "frc", "gnm", "mfd", "mfd_1", "mfd_2", "mfd_3", "mfd_4"],
}


def _pearson(x: list[float], y: list[float]) -> float:
    a, b = np.asarray(x, float), np.asarray(y, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def c3_descriptors(ctx: EvalContext) -> CheckResult:
    ref_dir = ctx.gold_dir / "reference_descriptors"
    details = []
    # Expected correlations from Fig. 2a, pooled over every chain the agents computed.
    pooled: dict[str, list[float]] = {c: [] for fam in FAMILIES.values() for c in fam}
    for fam, cols in FAMILIES.items():
        for f in sorted((ctx.workspace / "features" / fam).glob("*.csv")):
            for r in read_csv(f)[1]:
                vals = [_f(r.get(c)) for c in cols]
                if None not in vals:
                    for c, v in zip(cols, vals):
                        pooled[c].append(v)  # type: ignore[arg-type]
    corr = {}
    for a, b in (("orc", "frc"), ("shell", "rinacc")):
        if len(pooled[a]) == len(pooled[b]):
            corr[f"{a}~{b}"] = _pearson(pooled[a], pooled[b])
    corr_ok = [abs(v) >= 0.5 for v in corr.values() if not math.isnan(v)]
    details.append("Fig. 2a expected correlations: " + ", ".join(f"{k} r={v:.2f}" for k, v in corr.items()) if corr
                   else "Fig. 2a correlations: not enough feature data")

    if not ref_dir.exists():
        details.append("Reference descriptors not generated yet. Run scripts/make_reference_descriptors.py "
                       "with the capsule code to create mathbac/eval/gold/cha2022/reference_descriptors/.")
        return CheckResult("C3", "Descriptors correct", "not_scored", 0.0, details, {"correlations": corr})

    per_col: dict[str, list[float]] = {}
    for ref in sorted(ref_dir.glob("*.csv")):
        _, ref_rows = read_csv(ref)
        for fam, cols in FAMILIES.items():
            f = ctx.workspace / "features" / fam / ref.name
            if not f.exists():
                for c in cols:
                    per_col.setdefault(c, []).append(float("nan"))
                continue
            rows = {r["resnum"]: r for r in read_csv(f)[1]}
            for c in cols:
                if c not in ref_rows[0]:
                    continue
                pairs = [(_f(rr[c]), _f(rows.get(rr["resnum"], {}).get(c))) for rr in ref_rows]
                pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                per_col.setdefault(c, []).append(_pearson([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) >= 3 else float("nan"))
    col_ok = {c: bool(v) and all(not math.isnan(x) and x >= 0.99 for x in v) for c, v in per_col.items()}
    frac = sum(col_ok.values()) / len(col_ok) if col_ok else 0.0
    corr_frac = sum(corr_ok) / len(corr_ok) if corr_ok else 0.0
    score = 0.8 * frac + 0.2 * corr_frac
    details.append(f"{sum(col_ok.values())}/{len(col_ok)} descriptors match the reference (Pearson r ≥ 0.99 on every reference chain)")
    bad = [c for c, ok in col_ok.items() if not ok]
    if bad:
        details.append(f"not matching: {', '.join(bad)}")
    return CheckResult("C3", "Descriptors correct", "pass" if frac == 1.0 else "fail", score, details,
                       {"columns": col_ok, "correlations": corr})


# ---------------------------------------------------------------------------
def c4_table1(ctx: EvalContext) -> CheckResult:
    gold = ctx.gold("table1.yaml")
    tol = gold["tolerance"]
    pred_dir = ctx.workspace / "ml" / "cv" / "predictions"
    got: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    cells, ok_cells, missing = 0, 0, []
    rows = []
    for model, subsets in gold["table"].items():
        for subset, mets in subsets.items():
            f = pred_dir / f"{model}_{subset}.npz"
            cells += len(mets)
            if not f.exists():
                missing.append(f"{model}_{subset}")
                continue
            arr, errs = load_npz(f, ["fold", "y_true", "y_score"])
            if errs:
                missing.append(f"{model}_{subset}")
                continue
            res = metrics.per_fold(arr["fold"], arr["y_true"], arr["y_score"])
            got[(model, subset)] = res
            for m, (mean, _sd) in mets.items():
                good = abs(res[m][0] - mean) <= tol
                ok_cells += good
                rows.append({"model": model, "subset": subset, "metric": m, "paper": mean,
                             "reproduced": round(res[m][0], 4), "sd": round(res[m][1], 4), "within_tol": good})

    def auc(key: list[str]) -> float | None:
        r = got.get((key[0], key[1]))
        return r["auc"][0] if r else None

    order_ok: dict[str, bool | None] = {}
    for o in gold["orderings"]:
        if "left" in o:
            l, r = auc(o["left"]), auc(o["right"])
            order_ok[o["id"]] = None if l is None or r is None else l > r
        elif "strong" in o:
            s = [auc([m, o["subset"]]) for m in o["strong"]]
            w = [auc([m, o["subset"]]) for m in o["weak"]]
            order_ok[o["id"]] = None if None in s + w else min(s) > max(w)  # type: ignore[type-var]
        else:
            b, p = auc(o["base"]), auc(o["plus"])
            order_ok[o["id"]] = None if b is None or p is None else abs(p - b) <= o["max_abs_delta"]
    headline_ok = all(
        any(r["model"] == m and r["subset"] == s for r in rows) and
        all(r["within_tol"] for r in rows if r["model"] == m and r["subset"] == s)
        for m, s in gold["headline"]
    )
    orders = [v for v in order_ok.values() if v is not None]
    score = 0.5 * (ok_cells / cells) + 0.5 * (sum(orders) / len(order_ok) if order_ok else 0.0)
    status = "pass" if headline_ok and len(orders) == len(order_ok) and all(orders) else "fail"
    details = [f"{ok_cells}/{cells} Table 1 cells within ±{tol} of the paper",
               f"headline cells (DNN GT, GE_GT, all, CH; RF/XGB all) {'all' if headline_ok else 'not all'} within tolerance",
               "orderings: " + ", ".join(f"{k}={'yes' if v else 'no' if v is False else 'n/a'}" for k, v in order_ok.items())]
    if missing:
        details.append(f"no predictions for: {', '.join(missing[:12])}" + (" ..." if len(missing) > 12 else ""))
    return CheckResult("C4", "Table 1 reproduced", status, score, details, {"cells": rows, "orderings": order_ok})


# ---------------------------------------------------------------------------
def _interface_from_csv(path: Path, top_fraction: float) -> tuple[set[str], set[str]]:
    _, rows = read_csv(path)
    probs = np.array([_f(r["prob"]) or 0.0 for r in rows])
    n = max(int(len(rows) * top_fraction), 1)
    top = np.argsort(probs)[-n:]
    return {str(rows[i]["res_a"]) for i in top}, {str(rows[i]["res_b"]) for i in top}


def c5_heldout(ctx: EvalContext) -> CheckResult:
    gold = ctx.gold("heldout.yaml")
    spec_frac = 0.001
    results, details = [], []
    for cx in gold["complexes"]:
        base = ctx.workspace / "predictions" / "ppi" / f"{cx['pdb'].lower()}_{gold['model']}_{gold['subset']}"
        csv_p, json_p = base.with_suffix(".csv"), Path(str(base) + "_interface.json")
        if csv_p.exists():
            pa, pb = _interface_from_csv(csv_p, spec_frac)
            src = "recomputed from pair probabilities"
        elif json_p.exists():
            j = json.loads(json_p.read_text())
            pa, pb = {str(x) for x in j.get("chain_a", [])}, {str(x) for x in j.get("chain_b", [])}
            src = "from interface file"
        else:
            details.append(f"{cx['pdb']}: no predictions")
            results.append(None)
            continue
        try:
            a = pdb.load_chain(cx["pdb"], cx["chain_a"], ctx.pdb_cache)
            b = pdb.load_chain(cx["pdb"], cx["chain_b"], ctx.pdb_cache)
        except (OSError, ValueError) as e:
            details.append(f"{cx['pdb']}: ground truth unavailable ({e})")
            results.append(None)
            continue
        ta, tb = pdb.interface(a, b, gold["contact_cutoff_A"])
        sa = metrics.set_scores(pa, ta, set(a.resnums))
        sb = metrics.set_scores(pb, tb, set(b.resnums))
        acc = (sa["accuracy"] * len(a.resnums) + sb["accuracy"] * len(b.resnums)) / (len(a.resnums) + len(b.resnums))
        rec = (sa["tp"] + sb["tp"]) / max(sa["tp"] + sb["tp"] + sa["fn"] + sb["fn"], 1)
        prec = (sa["tp"] + sb["tp"]) / max(sa["tp"] + sb["tp"] + sa["fp"] + sb["fp"], 1)
        results.append(acc)
        details.append(f"{cx['pdb']}: residue accuracy {acc:.3f}, precision {prec:.2f}, recall {rec:.2f} ({src})")
    scored = [r for r in results if r is not None]
    passed = [r >= gold["pass_accuracy"] for r in scored]
    score = sum(passed) / len(gold["complexes"])
    status = "pass" if len(scored) == len(gold["complexes"]) and all(passed) else (
        "not_scored" if not scored and any("unavailable" in d for d in details) else "fail")
    return CheckResult("C5", "Held-out PPI interfaces", status, score, details, {"accuracy": results})


# ---------------------------------------------------------------------------
def _find_case(cases: list[dict], patterns: list[str]) -> dict | None:
    """First case whose name or PDB id contains any of the patterns."""
    for c in cases:
        text = f"{c.get('case', '')} {c.get('pdb_id', '')}".lower()
        if any(p.lower() in text for p in patterns):
            return c
    return None


def _corona(ws: Path) -> list[dict]:
    p = ws / "predictions" / "pni" / "corona_ranking.csv"
    return read_csv(p)[1] if p.exists() else []


def _score_for(rows: list[dict], np_name: str, protein: str) -> float | None:
    for r in rows:
        if np_name.lower() in r["np_name"].lower() and protein.lower() in r["protein"].lower():
            return _f(r["score"])
    return None


def c6_pni(ctx: EvalContext) -> CheckResult:
    gold = ctx.gold("pni_findings.yaml")
    cases_p = ctx.workspace / "data" / "np" / "cases.csv"
    cases = read_csv(cases_p)[1] if cases_p.exists() else []
    corona = _corona(ctx.workspace)
    outcome: dict[str, bool | None] = {}
    details = []
    for fd in gold["findings"]:
        fid, kind = fd["id"], fd["kind"]
        if kind == "review":
            v = ctx.reviews.get(fid)
            outcome[fid] = None if v is None else v == "pass"
            details.append(f"{fid}: {'reviewed ' + v if v else 'needs review'} - {fd['desc']}")
            continue
        if kind == "ranking":
            if "less_than" in fd:
                checks = []
                for (np1, p1), (np2, p2) in fd["less_than"]:
                    s1, s2 = _score_for(corona, np1, p1), _score_for(corona, np2, p2)
                    checks.append(None if s1 is None or s2 is None else s1 < s2)
            else:
                prots = {r["protein"] for r in corona if fd["higher_np"].lower() in r["np_name"].lower()}
                checks = [None if (lo := _score_for(corona, fd["lower_np"], p)) is None or (hi := _score_for(corona, fd["higher_np"], p)) is None
                          else lo < hi for p in prots] or [None]
            outcome[fid] = None if None in checks else all(checks)
            details.append(f"{fid}: {outcome[fid]} - {fd['desc']}")
            continue
        case = _find_case(cases, fd["case_match"])
        j = ctx.workspace / "predictions" / "pni" / f"{case['case']}_interface.json" if case else None
        if not case or not j or not j.exists():
            outcome[fid] = False
            details.append(f"{fid}: no prediction for this case - {fd['desc']}")
            continue
        pred = [str(x) for x in json.loads(j.read_text()).get("protein_residues", [])]
        try:
            chain = pdb.load_chain(case["pdb_id"], case["chain"], ctx.pdb_cache)
        except (OSError, ValueError) as e:
            outcome[fid] = None
            details.append(f"{fid}: ground truth unavailable ({e})")
            continue
        if kind == "residues":
            labels = {chain.label(i) for i, rn in enumerate(chain.resnums) if rn in pred}
            hits = sorted(labels & set(fd["residues"]))
            outcome[fid] = len(hits) >= fd["min_hits"]
            details.append(f"{fid}: {len(hits)}/{len(fd['residues'])} expected residues predicted ({', '.join(hits) or 'none'})")
        elif kind == "n_terminal":
            idx = [chain.resnums.index(r) for r in pred if r in chain.resnums]
            share = (sum(i < fd["fraction"] * len(chain.resnums) for i in idx) / len(idx)) if idx else 0.0
            outcome[fid] = share >= fd["min_share"]
            details.append(f"{fid}: {share:.0%} of predicted residues in the N-terminal third")
    passed = sum(1 for v in outcome.values() if v)
    pending = sum(1 for v in outcome.values() if v is None)
    score = passed / len(outcome)
    if passed >= gold["pass_count"]:
        status = "pass"
    elif passed + pending >= gold["pass_count"]:
        status = "needs_review"
    else:
        status = "fail"
    details.insert(0, f"{passed}/{len(outcome)} findings matched, {pending} pending review (pass at {gold['pass_count']})")
    return CheckResult("C6", "Protein-NP predictions", status, score, details, {"findings": outcome})


# ---------------------------------------------------------------------------
RUBRIC = ["fidelity", "execution", "accuracy", "completeness", "rigor"]


def c7_report(ctx: EvalContext) -> CheckResult:
    p = ctx.run_dir / "eval" / "c7_scores.yaml"
    if not (ctx.workspace / "report" / "report.md").exists():
        return CheckResult("C7", "Report and claim verdicts", "fail", 0.0, ["report/report.md missing"])
    if not p.exists():
        return CheckResult("C7", "Report and claim verdicts", "needs_review", 0.0,
                           [f"Score the report 1-5 on {', '.join(RUBRIC)} in eval/c7_scores.yaml (by a reviewer or `mathbac judge`)."])
    s = yaml.safe_load(p.read_text()) or {}
    vals = {k: s.get(k) for k in RUBRIC}
    got = [v for v in vals.values() if isinstance(v, (int, float))]
    mean = sum(got) / len(got) if got else 0.0
    return CheckResult("C7", "Report and claim verdicts", "pass" if len(got) == len(RUBRIC) else "needs_review",
                       mean / 5, [", ".join(f"{k} {v}" for k, v in vals.items())], {"rubric": vals, "by": s.get("by")})


ALL: list[Callable[[EvalContext], CheckResult]] = [
    c0_environment, c1_spec, c2_data, c3_descriptors, c4_table1, c5_heldout, c6_pni, c7_report,
]
