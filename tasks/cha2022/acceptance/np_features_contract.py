"""N4d: GE+GT descriptor files for every nanoparticle, rows aligned with the coarse-grained nodes."""

from pathlib import Path

from mathbac.contracts import limit, read_csv, require_columns

COLUMNS = ["rinacc", "shell", "pocketness", "opd_5", "opd_7", "orc", "frc", "gnm", "mfd"]


def check(workspace: Path, node) -> list[str]:
    np_dir = workspace / "data" / "np"
    problems: list[str] = []
    for cg in sorted(np_dir.glob("*_cg.csv")):
        name = cg.name[: -len("_cg.csv")]
        f = workspace / "features" / "np" / f"{name}.csv"
        cols = require_columns(f, ["node", *COLUMNS])
        if cols:
            problems += cols
            continue
        if len(read_csv(f)[1]) != len(read_csv(cg)[1]):
            problems.append(f"{f.name}: row count differs from {cg.name}")
    return limit(problems)
