"""N4a: CH descriptors per residue, following SI 2.1 and the released code.

- Partial charges: PDB2PQR with the CHARMM force field (SI 2.1.1), summed per residue by element
  (n_charge, c_charge, o_charge, h_charge, s_charge); atom counts by element after PDB2PQR adds hydrogens.
- Hydrophobicity: Kyte-Doolittle (SI 2.1.3; released code hydrophobicity_fill).
- Molecular weight: the residue masses in the released local_OPD.MW table.
- Class flags from the released code's binary_fill: pos, neg, polar, amp, hp; `polarity` = the polar flag.

    python3 scripts/n4_ch.py [--procs 4]     # needs the `pdb2pqr` command (pip install pdb2pqr)
"""

import argparse
import csv
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

KD = {"ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5, "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7,
      "SER": -0.8, "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5, "GLN": -3.5, "ASP": -3.5,
      "ASN": -3.5, "LYS": -3.9, "ARG": -4.5}
MW = {"ALA": 89.1, "ARG": 174.2, "ASN": 132.1, "ASP": 133.1, "CYS": 121.2, "GLU": 147.1, "GLN": 146.2, "GLY": 75.1,
      "HIS": 155.2, "ILE": 131.2, "LEU": 131.2, "LYS": 146.2, "MET": 149.2, "PHE": 165.2, "PRO": 115.1, "SER": 105.1,
      "THR": 119.1, "TRP": 204.2, "TYR": 181.2, "VAL": 117.1}
POS, NEG = {"ARG", "LYS", "HIS"}, {"ASP", "GLU"}
POLAR = {"GLN", "ASN", "SER", "THR", "TYR", "CYS"}
AMP = {"TRP", "TYR", "MET"}
HP = {"ALA", "ILE", "LEU", "MET", "PHE", "VAL", "PRO", "GLY"}
ELEMENTS = ["N", "C", "O", "H", "S"]


def chains():
    out = []
    for r in csv.DictReader(open("data/pairs.csv")):
        for ch in (r["chain_a"], r["chain_b"]):
            if (r["pdb_id"], ch) not in out:
                out.append((r["pdb_id"], ch))
    return out


def run(job):
    pid, ch = job
    out = Path(f"features/CH/{pid.lower()}_{ch}.csv")
    if out.exists():
        return job, "cached"
    ca = list(csv.DictReader(open(f"data/chains/{pid.lower()}_{ch}.csv")))
    keep = {r["resnum"] for r in ca}
    with tempfile.TemporaryDirectory() as t:
        src = Path(t) / "in.pdb"
        lines = []
        for line in open(f"data/pdb/{pid.upper()}.pdb", errors="replace"):
            if line.startswith("ENDMDL"):
                break
            if line.startswith("ATOM") and line[21] == ch and line[16] in " A" and line[22:27].strip() in keep:
                lines.append(line[:16] + " " + line[17:])
        src.write_text("".join(lines) + "END\n")
        pqr = Path(t) / "out.pqr"
        p = subprocess.run(["pdb2pqr", "--ff=CHARMM", "--keep-chain", str(src), str(pqr)], capture_output=True, text=True)
        charge = {}
        if pqr.exists():
            for line in pqr.read_text().splitlines():
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                parts = line[30:].split()
                q = float(parts[3])
                name = line[12:16].strip()
                el = name[0] if name[0] in "NCOHS" else ("H" if name[0].isdigit() else name[0])
                rn = line[22:27].strip()
                d = charge.setdefault(rn, {f"{e}_{k}": 0.0 for e in ELEMENTS for k in ("charge", "count")})
                if el in ELEMENTS:
                    d[f"{el}_charge"] += q
                    d[f"{el}_count"] += 1
    rows = []
    missing = 0
    for r in ca:
        n = r["resname"]
        c = charge.get(r["resnum"])
        if c is None:
            missing += 1
            c = {f"{e}_{k}": float("nan") for e in ELEMENTS for k in ("charge", "count")}
        rows.append([r["resnum"], n, round(c["C_charge"], 4), KD.get(n, 0.0), int(n in POLAR), MW.get(n, 120.0),
                     int(n in POS), int(n in NEG), int(n in AMP), int(n in HP),
                     *[round(c[f"{e}_charge"], 4) for e in ["N", "O", "H", "S"]], *[c[f"{e}_count"] for e in ELEMENTS]])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["resnum", "resname", "c_charge", "hydrophobicity", "polarity", "mw", "pos", "neg", "amp", "hp",
                    "n_charge", "o_charge", "h_charge", "s_charge", "n_count", "c_count", "o_count", "h_count", "s_count"])
        w.writerows(rows)
    return job, f"done ({missing} residues without PDB2PQR charges)" if missing else "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    jobs = chains()
    notes = []
    with ProcessPoolExecutor(a.procs) as ex:
        for k, (job, state) in enumerate(ex.map(run, jobs), 1):
            if "without" in state:
                notes.append(f"{job}: {state}")
    print(f"CH descriptors for {len(jobs)} chains; {len(notes)} chains with residues PDB2PQR dropped")
    for n in notes[:20]:
        print("  ", n)


if __name__ == "__main__":
    main()
