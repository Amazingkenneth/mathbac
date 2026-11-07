"""N9: report exists and claims.yaml gives a verdict with evidence for every claim."""

from pathlib import Path

from mathbac.contracts import load_yaml

VERDICTS = {"supported", "inconclusive", "contradicted"}
MIN_CLAIMS = 5  # the task brief lists five claims


def check(workspace: Path, node) -> list[str]:
    problems = []
    report = workspace / "report" / "report.md"
    if report.exists() and len(report.read_text().split()) < 300:
        problems.append("report.md is shorter than 300 words")
    try:
        claims = load_yaml(workspace / "report" / "claims.yaml")
    except Exception as e:
        return problems + [f"claims.yaml is not valid YAML: {e}"]
    if not isinstance(claims, list) or len(claims) < MIN_CLAIMS:
        return problems + [f"claims.yaml must list at least {MIN_CLAIMS} claims"]
    for i, c in enumerate(claims):
        if c.get("verdict") not in VERDICTS:
            problems.append(f"claim {i}: verdict must be one of {sorted(VERDICTS)}")
        for p in c.get("evidence") or []:
            if not (workspace / p).exists():
                problems.append(f"claim {i}: evidence path {p} does not exist")
        if not c.get("evidence"):
            problems.append(f"claim {i}: no evidence paths")
    return problems
