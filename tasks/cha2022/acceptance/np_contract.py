"""N3c: coarse-grained nanoparticle files and the case list."""

from pathlib import Path

from mathbac.contracts import limit, read_csv, require_columns


def check(workspace: Path, node) -> list[str]:
    np_dir = workspace / "data" / "np"
    problems = require_columns(np_dir / "cases.csv", ["case", "np_name", "pdb_id", "chain", "reference"])
    if problems:
        return problems
    _, cases = read_csv(np_dir / "cases.csv")
    for name in sorted({c["np_name"] for c in cases}):
        f = np_dir / f"{name}_cg.csv"
        cols = require_columns(f, ["node", "x", "y", "z", "n_atoms", "element"])
        if cols:
            problems += cols
            continue
        _, rows = read_csv(f)
        if not rows:
            problems.append(f"{f.name} has no nodes")
    return limit(problems)
