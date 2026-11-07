"""N7: held-out PPI predictions and interface files for at least the DNN GE_GT model.

The interface file must be the top `top_fraction` of the pair CSV, with every pair tied with the last one included,
so that anyone can rebuild it from the CSV. Ties beyond a handful mean the probabilities lost precision (a saturated
DNN written with six decimals gives many pairs exactly 1): write full precision, e.g. repr(float(p)), computing
P(contact) as 1 - mean P(non-contact) in float64.
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
    probs = sorted(r[0] for r in rows)
    n = max(int(len(rows) * float(iface.get("top_fraction", 0.001))), 1)
    cut = probs[-n]
    top = [r for r in rows if r[0] >= cut]
    problems = []
    if len(top) - n > max(2, n // 10):
        problems.append(f"{csv_path.name}: {len(top) - n} extra pairs tie at the top-{n} cut (prob {cut}); the probabilities "
                        "lost precision. Write full precision (repr(float(p)); P(contact) as 1 - mean P(non-contact) in float64)")
    a, b = {r[1] for r in top}, {r[2] for r in top}
    if a != {str(x) for x in iface["chain_a"]} or b != {str(x) for x in iface["chain_b"]}:
        problems.append(f"{csv_path.name}: the top {n} pairs (ties at the cut included) do not give the residues in the interface file")
    return problems
