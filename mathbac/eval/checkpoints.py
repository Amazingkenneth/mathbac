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
    "CH": ["c_charge", "hydrophobicity", "polarity", "mw"],
    "GE": ["rinacc", "shell", "pocketness", "opd_5", "opd_7", "opd_10", "opd_15"],
    "GT": ["orc", "frc", "gnm", "mfd", "mfd_1", "mfd_2", "mfd_3", "mfd_4"],
}
# Fig. 2a names -> contract column names (OPD is skipped: the figure does not say which neighbourhood size)
FIG2_COLUMNS = {"MW": "mw", "Hp": "hydrophobicity", "C_charge": "c_charge", "C_count": "c_count", "Shell": "shell",
                "Rinacc": "rinacc", "Pocket": "pocketness", "ORC": "orc", "FRC": "frc", "GNM": "gnm", "MFD": "mfd"}


def _pearson(x: list[float], y: list[float]) -> float:
    a, b = np.asarray(x, float), np.asarray(y, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _chain_features(ws: Path, stem: str) -> dict[str, dict[str, float]]:
    """resnum -> {column: value} merged over the CH, GE and GT files of one chain."""
    rows: dict[str, dict[str, float]] = {}
    for fam in FAMILIES:
        f = ws / "features" / fam / f"{stem}.csv"
        if not f.exists():
            continue
        for r in read_csv(f)[1]:
            d = rows.setdefault(str(r["resnum"]), {})
            for k, v in r.items():
                if k not in ("resnum", "resname") and (x := _f(v)) is not None:
                    d[k] = x
    return rows


def c3_descriptors(ctx: EvalContext) -> CheckResult:
    ref_dir = ctx.gold_dir / "reference_descriptors"
    cfg_p = ctx.gold_dir / "descriptor_check.yaml"
    cfg = yaml.safe_load(cfg_p.read_text()) if cfg_p.exists() else {}
    min_r = float(cfg.get("min_r", 0.99))
    overrides = cfg.get("min_r_overrides", {})
    details: list[str] = []

    # 1. Fig. 2a cross-correlations, pooled over every chain the agents computed.
    stems = sorted({f.stem for fam in FAMILIES for f in (ctx.workspace / "features" / fam).glob("*.csv")})
    pooled = [row for s in stems for row in _chain_features(ctx.workspace, s).values()]
    fig2 = ctx.gold("fig2a_correlations.yaml")["matrix"] if (ctx.gold_dir / "fig2a_correlations.yaml").exists() else {}
    corr_rows = []
    names = [n for n in fig2 if n in FIG2_COLUMNS]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            target = fig2[a].get(b)
            ca, cb = FIG2_COLUMNS[a], FIG2_COLUMNS[b]
            if target is None or abs(target) < 0.3:
                continue
            xy = [(r[ca], r[cb]) for r in pooled if ca in r and cb in r]
            got = _pearson([p[0] for p in xy], [p[1] for p in xy]) if len(xy) >= 30 else float("nan")
            ok = not math.isnan(got) and np.sign(got) == np.sign(target) and abs(got - target) <= 0.25
            corr_rows.append({"pair": f"{ca}~{cb}", "paper": target, "reproduced": None if math.isnan(got) else round(got, 3), "ok": ok})
    corr_frac = sum(r["ok"] for r in corr_rows) / len(corr_rows) if corr_rows else 0.0
    details.append(f"Fig. 2a correlations (|r| >= 0.3 in the paper): {sum(r['ok'] for r in corr_rows)}/{len(corr_rows)} reproduced "
                   f"within ±0.25 with the same sign, over {len(pooled)} residues")

    # 2. Per-residue values against the authors' reference chains (Fig. 1 source data for 1MA9 A).
    refs = sorted(ref_dir.glob("*.csv")) if ref_dir.exists() else []
    if not refs:
        details.append("No reference descriptors in the gold folder (run scripts/build_gold_from_source_data.py).")
        return CheckResult("C3", "Descriptors correct", "not_scored", 0.0, details, {"correlations": corr_rows})
    col_r: dict[str, float] = {}
    for ref in refs:
        header, ref_rows = read_csv(ref)
        mine = _chain_features(ctx.workspace, ref.stem)
        for c in [h for h in header if h not in ("resnum", "resname")]:
            candidates = [c]
            best = float("nan")
            for cand in candidates:
                xy = [(_f(rr[c]), mine.get(rr["resnum"], {}).get(cand)) for rr in ref_rows]
                xy = [(x, y) for x, y in xy if x is not None and y is not None]
                r = _pearson([p[0] for p in xy], [p[1] for p in xy]) if len(xy) >= max(3, len(ref_rows) // 2) else float("nan")
                if not math.isnan(r) and (math.isnan(best) or r > best):
                    best = r
            key = f"{ref.stem}:{c}"
            col_r[key] = best
    col_ok = {k: not math.isnan(v) and v >= float(overrides.get(k.split(":")[1], min_r)) for k, v in col_r.items()}
    frac = sum(col_ok.values()) / len(col_ok)
    score = 0.8 * frac + 0.2 * corr_frac
    details.append(f"{sum(col_ok.values())}/{len(col_ok)} reference descriptors match (Pearson r >= {min_r}, per-column overrides "
                   f"{overrides or 'none'})")
    details.append("r by descriptor: " + ", ".join(f"{k.split(':')[1]} {'n/a' if math.isnan(v) else f'{v:.3f}'}" for k, v in col_r.items()))
    status = "pass" if frac == 1.0 and corr_frac >= 0.8 else "fail"
    return CheckResult("C3", "Descriptors correct", status, score, details,
                       {"columns": {k: (None if math.isnan(v) else round(v, 4)) for k, v in col_r.items()}, "correlations": corr_rows})



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
            res = metrics.per_fold(arr["fold"], arr["y_true"], arr["y_score"], f1_class=gold.get("f1_class", "contact"))
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
def _interface_from_csv(path: Path, top_fraction: float) -> tuple[set[str], set[str], int]:
    """Top-fraction residues from a pair CSV, plus the number of pairs tied at the cut (0 when the cut is clean)."""
    _, rows = read_csv(path)
    probs = np.array([_f(r["prob"]) or 0.0 for r in rows])
    n = max(int(len(rows) * top_fraction), 1)
    order = np.argsort(probs, kind="mergesort")
    top = order[-n:]
    cut = probs[top[0]]
    ties = int(np.sum(probs == cut)) if n < len(rows) and probs[order[-n - 1]] == cut else 0
    return {str(rows[i]["res_a"]) for i in top}, {str(rows[i]["res_b"]) for i in top}, ties


def c5_heldout(ctx: EvalContext) -> CheckResult:
    gold = ctx.gold("heldout.yaml")
    authors_p = ctx.gold_dir / "heldout_authors.yaml"
    authors = yaml.safe_load(authors_p.read_text()) if authors_p.exists() else {}
    results: list[float | None] = []
    passed: list[bool] = []
    details: list[str] = []
    for cx in gold["complexes"]:
        base = ctx.workspace / "predictions" / "ppi" / f"{cx['pdb'].lower()}_{gold['model']}_{gold['subset']}"
        csv_p, json_p = base.with_suffix(".csv"), Path(str(base) + "_interface.json")
        if csv_p.exists():
            pa, pb, ties = _interface_from_csv(csv_p, gold["top_fraction"])
            src = "recomputed from pair probabilities" + (f"; WARNING {ties} pairs tie at the cut, so the set is arbitrary" if ties else "")
        elif json_p.exists():
            j = json.loads(json_p.read_text())
            pa, pb = {str(x) for x in j.get("chain_a", [])}, {str(x) for x in j.get("chain_b", [])}
            src = "from interface file"
        else:
            details.append(f"{cx['pdb']}: no predictions")
            results.append(None)
            passed.append(False)
            continue
        try:
            a = pdb.load_chain(cx["pdb"], cx["chain_a"], ctx.pdb_cache)
            b = pdb.load_chain(cx["pdb"], cx["chain_b"], ctx.pdb_cache)
        except (OSError, ValueError) as e:
            details.append(f"{cx['pdb']}: ground truth unavailable ({e})")
            results.append(None)
            passed.append(False)
            continue
        ta, tb = pdb.interface(a, b, gold["contact_cutoff_A"])
        sa = metrics.set_scores(pa, ta, set(a.resnums))
        sb = metrics.set_scores(pb, tb, set(b.resnums))
        tp, fp, fn = sa["tp"] + sb["tp"], sa["fp"] + sb["fp"], sa["fn"] + sb["fn"]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        target = cx["authors_f1"] - gold["f1_margin"]
        ok = f1 >= target
        results.append(f1)
        passed.append(ok)
        line = f"{cx['pdb']}: per-residue F1 {f1:.3f} (precision {prec:.2f}, recall {rec:.2f}; authors {cx['authors_f1']:.3f}, pass >= {target:.3f}; {src})"
        ap = (authors.get(cx["pdb"]) or {}).get("dnn_GE_GT_predicted")
        if ap:
            ours, theirs = {("A", x) for x in pa} | {("B", x) for x in pb}, {("A", x) for x in ap["A"]} | {("B", x) for x in ap["B"]}
            jac = len(ours & theirs) / len(ours | theirs) if ours | theirs else 0.0
            line += f"; overlap with the authors' predicted residues (Jaccard) {jac:.2f}"
        details.append(line)
    score = sum(passed) / len(gold["complexes"])
    if all(passed):
        status = "pass"
    elif any("unavailable" in d for d in details) and not any(r is not None for r in results):
        status = "not_scored"
    else:
        status = "fail"
    details.append("SI Tables 4-6 report accuracy 0.721 / 0.803 / 0.786 under a definition the published data do not allow us to reconstruct.")
    return CheckResult("C5", "Held-out PPI interfaces", status, score, details, {"f1": results})


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
    """Mean score over rows whose NP and protein names contain the given strings (several SWNT types average)."""
    vals = [_f(r["score"]) for r in rows if np_name.lower() in r["np_name"].lower() and protein.lower() in r["protein"].lower()]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _residue_labels(pdb_id: str, chains: list[str], residues: list[str], cache: Path) -> set[str]:
    labels = set()
    for ch in chains:
        c = pdb.load_chain(pdb_id, ch, cache)
        for i, rn in enumerate(c.resnums):
            if rn in residues:
                labels.add(c.label(i))
    return labels


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
            details.append(f"{fid}: {'reviewed ' + v if v else 'needs review'} - {fd['desc'].strip()}")
            continue
        if kind == "ranking":
            if "less_than" in fd:
                checks = []
                for (np1, p1), (np2, p2) in fd["less_than"]:
                    s1, s2 = _score_for(corona, np1, p1), _score_for(corona, np2, p2)
                    checks.append(None if s1 is None or s2 is None else s1 < s2)
            else:
                prots = sorted({r["protein"] for r in corona if fd["higher_np"].lower() in r["np_name"].lower()})
                checks = []
                for prot in prots:
                    lo, hi = _score_for(corona, fd["lower_np"], prot), _score_for(corona, fd["higher_np"], prot)
                    checks.append(None if lo is None or hi is None else lo < hi)
                checks = checks or [None]
            outcome[fid] = False if None in checks else all(checks)
            details.append(f"{fid}: {'yes' if outcome[fid] else 'no'} - {fd['desc'].strip()}")
            continue
        case = _find_case(cases, fd["case_match"])
        j = ctx.workspace / "predictions" / "pni" / f"{case['case']}_interface.json" if case else None
        if not case or not j or not j.exists():
            outcome[fid] = False
            details.append(f"{fid}: no prediction for this case")
            continue
        pred = [str(x) for x in json.loads(j.read_text()).get("protein_residues", [])]
        if kind == "residues":
            try:
                labels = _residue_labels(case["pdb_id"], fd.get("chains") or [case["chain"]], pred, ctx.pdb_cache)
            except (OSError, ValueError) as e:
                outcome[fid] = None
                details.append(f"{fid}: structure unavailable ({e})")
                continue
            hits = sorted(labels & set(fd["required"]))
            named = sorted(labels & set(fd.get("reported", [])))
            outcome[fid] = len(hits) == len(fd["required"])
            details.append(f"{fid}: required {', '.join(fd['required'])}: {len(hits)} predicted; residues named in the text "
                           f"that were predicted: {', '.join(named) or 'none'}")
        elif kind == "overlap":
            lo, hi = fd["region"]
            nums = [int(x) for x in pred if str(x).lstrip("-").isdigit()]
            share = sum(lo <= n <= hi for n in nums) / len(nums) if nums else 0.0
            outcome[fid] = share >= fd["min_share"]
            details.append(f"{fid}: {share:.0%} of predicted residues in {lo}-{hi} (pass at {fd['min_share']:.0%})")
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
