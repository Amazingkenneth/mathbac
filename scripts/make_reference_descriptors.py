"""Generate reference per-residue descriptors for checkpoint C3 from the released code.

Run inside the sandbox image (it needs the capsule's Python stack and GHECOM):

    docker run --rm -v "$PWD":/repo -w /repo mathbac-cha2022:latest \
        python scripts/make_reference_descriptors.py --out mathbac/eval/gold/cha2022/reference_descriptors

For each reference chain it writes <pdb>_<chain>.csv with resnum, resname and every CH, GE and GT
column of the artifact contract, computed by the released functions through env/tools/descriptors.py
and env/tools/ghecom.sh. The output is gold data: commit it, and never mount it into agent sandboxes.

CH columns and the GHECOM-derived GE columns come from the released PPI pipeline's descriptor
matrix; mapping those matrix columns onto the contract names is the one step still to finish here
(see TODO below), because the pipeline writes one combined matrix per complex.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
import urllib.request
from pathlib import Path

REFERENCE_CHAINS = [("6wzo", "A"), ("6wzo", "B"), ("1ma9", "A"), ("1ma9", "B")]
TOOLS = Path("/opt/mathbac/tools")


def ca_csv(pdb_path: Path, chain: str, out: Path) -> None:
    seen, rows = set(), []
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == chain and line[16] in " A":
            rn = line[22:27].strip()
            if rn not in seen:
                seen.add(rn)
                rows.append([rn, line[17:20].strip(), line[30:38].strip(), line[38:46].strip(), line[46:54].strip()])
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["resnum", "resname", "x", "y", "z"])
        w.writerows(rows)


def merge(paths: list[Path], out: Path) -> None:
    tables = []
    for p in paths:
        with p.open() as f:
            tables.append(list(csv.DictReader(f)))
    n = len(tables[0])
    if any(len(t) != n for t in tables):
        raise SystemExit(f"row counts differ while merging into {out}: {[len(t) for t in tables]}")
    cols = ["resnum", "resname"] + [c for t in tables for c in t[0] if c not in ("resnum", "resname", "index")]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for i in range(n):
            row = {}
            for t in tables:
                row.update(t[i])
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        for pdb, chain in REFERENCE_CHAINS:
            pdb_path = t / f"{pdb}.pdb"
            if not pdb_path.exists():
                urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb.upper()}.pdb", pdb_path)
            ca = t / f"{pdb}_{chain}_ca.csv"
            ca_csv(pdb_path, chain, ca)
            parts = []
            for kind in ("curvature", "mfd", "opd"):
                o = t / f"{pdb}_{chain}_{kind}.csv"
                subprocess.run(["python", str(TOOLS / "descriptors.py"), kind, "--ca", str(ca), "--out", str(o)], check=True)
                parts.append(o)
            # TODO: add GNM (align ProDy rows to ca.csv by residue number), GHECOM columns
            # (rinacc, shell, pocketness) and CH columns from the released pipeline's matrix.
            merge(parts, out / f"{pdb}_{chain}.csv")
            print(f"wrote {out / f'{pdb}_{chain}.csv'}")


if __name__ == "__main__":
    main()
