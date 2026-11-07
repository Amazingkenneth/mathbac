"""N8: an interface file for every protein-nanoparticle case, plus the corona ranking."""

from pathlib import Path

from mathbac.contracts import load_json, read_csv, require_columns


def check(workspace: Path, node) -> list[str]:
    problems = []
    cases_csv = workspace / "data" / "np" / "cases.csv"
    if cases_csv.exists():
        for c in read_csv(cases_csv)[1]:
            if "corona" in c["case"]:
                continue
            j = workspace / "predictions" / "pni" / f"{c['case']}_interface.json"
            if not j.exists():
                problems.append(f"missing predictions/pni/{j.name}")
            elif not load_json(j).get("protein_residues"):
                problems.append(f"{j.name}: protein_residues is empty")
    problems += require_columns(workspace / "predictions" / "pni" / "corona_ranking.csv", ["np_name", "protein", "score"])
    return problems
