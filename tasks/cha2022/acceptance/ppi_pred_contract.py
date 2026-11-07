"""N7: held-out PPI predictions and interface files for at least the DNN GE_GT model."""

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
    return problems
