"""N4a-c: one feature file per chain, required columns present, rows aligned with data/chains."""

import math
from pathlib import Path

from mathbac.contracts import chain_files, limit, read_csv, require_columns

REQUIRED = {
    "CH": ["c_charge", "hydrophobicity", "polarity", "mw"],
    "GE": ["rinacc", "shell", "pocketness", "opd_5", "opd_7", "opd_10", "opd_15"],
    "GT": ["orc", "frc", "gnm", "mfd", "mfd_1", "mfd_2", "mfd_3", "mfd_4"],
}


def check(workspace: Path, node) -> list[str]:
    family = node.facts.get("descriptor_family")
    if family not in REQUIRED:
        return [f"node facts must set descriptor_family to one of {sorted(REQUIRED)}"]
    problems: list[str] = []
    for pdb, ch in chain_files(workspace):
        ref = workspace / "data" / "chains" / f"{pdb}_{ch}.csv"
        f = workspace / "features" / family / f"{pdb}_{ch}.csv"
        cols = require_columns(f, ["resnum", "resname", *REQUIRED[family]])
        if cols:
            problems += cols
            continue
        _, rows = read_csv(f)
        _, ref_rows = read_csv(ref)
        if [r["resnum"] for r in rows] != [r["resnum"] for r in ref_rows]:
            problems.append(f"{f.name}: residue rows do not match data/chains/{ref.name} (count {len(rows)} vs {len(ref_rows)})")
            continue
        bad = 0
        for r in rows:
            for c in REQUIRED[family]:
                try:
                    if not math.isfinite(float(r[c])):
                        bad += 1
                except (TypeError, ValueError):
                    bad += 1
        if bad:
            problems.append(f"{f.name}: {bad} non-numeric or non-finite descriptor values")
    return limit(problems)
