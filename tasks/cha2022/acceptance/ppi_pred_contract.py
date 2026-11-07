"""N7: held-out PPI predictions and interface files for at least the DNN GE_GT model.

The interface file must be the top `top_fraction` of the pair CSV, and the CSV's probabilities must separate the
pairs at that cut: written with too few digits, a saturated DNN gives many pairs a probability of exactly 1, and
the top 0.1% is then an arbitrary choice among ties that nobody downstream can reproduce.
"""

import csv
from pathlib import Path

from mathbac.contracts import load_json, require_columns


def check(workspace: Path, node) -> list[str]:
    d = workspace / "predictions" / "ppi"
    problems = []
    for pdb in [p.lower() for p in node.facts.get("heldout_ppi", ["6wzo", "6nqn", "1c29"])]:
        base = d / f"{pdb}_dnn_GE_GT"
        problems += require_columns(base.with_suffix(".csv"), ["res_a", "res_b", "prob"])
        j = Path(str(base) + "_interface.json")
        if not j.exists():
            problems.append(f"missing {j.relative_to(workspace)}")
            continue
        iface = load_json(j)
        if not iface.get("chain_a") or not iface.get("chain_b"):
            problems.append(f"{j.name}: chain_a and chain_b must list predicted interface residues")
            continue
        problems += _consistent(base.with_suffix(".csv"), iface)
    return problems


def _consistent(csv_path: Path, iface: dict) -> list[str]:
    if not csv_path.exists():
        return []
    with csv_path.open() as fh:
        rows = [(float(r["prob"]), str(r["res_a"]), str(r["res_b"])) for r in csv.DictReader(fh)]
    if not rows:
        return [f"{csv_path.name}: no rows"]
    rows.sort(key=lambda r: r[0])
    n = max(int(len(rows) * float(iface.get("top_fraction", 0.001))), 1)
    top = rows[-n:]
    problems = []
    if n < len(rows) and rows[-n - 1][0] == top[0][0]:
        ties = sum(1 for r in rows if r[0] == top[0][0])
        problems.append(f"{csv_path.name}: {ties} pairs tie at the top-{n} cut (prob {top[0][0]}); write probabilities "
                        "with full precision (e.g. repr(float(p)), or 1 - P(non-contact) in float64)")
    a, b = {r[1] for r in top}, {r[2] for r in top}
    if a != {str(x) for x in iface["chain_a"]} or b != {str(x) for x in iface["chain_b"]}:
        problems.append(f"{csv_path.name}: the top {n} pairs do not give the residues in the interface file")
    return problems
