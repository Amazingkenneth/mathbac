"""N3, part 2: download structures and write the dataset in the artifact-contract format.

For each candidate entry, choose one chain of each entity, picking the chain pair with the most
inter-chain Cα pairs under 7 Å (a stand-in for the paper's PISA-verified interface), keep it if it has
>= 10 such pairs and both chains have 50-600 Cα residues. Residues: ATOM records with a Cα, first model,
altloc blank or A (as the released code reads PDB files with biopandas df['ATOM']).

    python scripts/n3_build.py --n 200
"""

import argparse
import csv
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PDB_DIR = Path("data/pdb")
ALWAYS_TRAIN = [("1MA9", "A", "B")]          # in the paper's training set (Fig. 1)
HELDOUT = [("6WZO", "A", "B"), ("6NQN", "A", "B"), ("1C29", "A", "B")]


def fetch(pid):
    p = PDB_DIR / f"{pid}.pdb"
    if p.exists() and p.stat().st_size > 0:
        return p
    for k in range(4):
        try:
            with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.pdb", timeout=60) as r:
                p.write_bytes(r.read())
            return p
        except OSError:
            time.sleep(2 ** k)
    return None


def chains_ca(path):
    out = {}
    seen = set()
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM") or line[12:16].strip() != "CA" or line[16] not in " A":
            continue
        ch, rn = line[21], line[22:27].strip()
        if (ch, rn) in seen:
            continue
        seen.add((ch, rn))
        out.setdefault(ch, []).append((rn, line[17:20].strip(), float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return out


def dist(a, b):
    A = np.array([r[2:] for r in a]); B = np.array([r[2:] for r in b])
    return np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))


def write_chain(pid, ch, rows):
    with (Path("data/chains") / f"{pid.lower()}_{ch}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["resnum", "resname", "x", "y", "z"])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    for d in ("data/pdb", "data/chains", "data/distances"):
        Path(d).mkdir(parents=True, exist_ok=True)
    cands = list(csv.DictReader(open("data/curation/candidate_pairs.csv")))
    cands = [c for c in cands if c["uniprot1"] != c["uniprot2"]]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(fetch, [c["pdb_id"] for c in cands[: a.n * 2]] + [p for p, _, _ in ALWAYS_TRAIN + HELDOUT]))

    chosen, rejected = [], []
    for pid, ca, cb in ALWAYS_TRAIN:
        chosen.append((pid, ca, cb, "train"))
    for c in cands:
        if len([x for x in chosen if x[3] == "train"]) >= a.n:
            break
        p = PDB_DIR / f"{c['pdb_id']}.pdb"
        if not p.exists():
            rejected.append((c["pdb_id"], "download failed")); continue
        chs = chains_ca(p)
        best = None
        for x in c["chains_entity1"].split():
            for y in c["chains_entity2"].split():
                if x in chs and y in chs and 50 <= len(chs[x]) <= 600 and 50 <= len(chs[y]) <= 600:
                    n = int((dist(chs[x], chs[y]) < 7).sum())
                    if best is None or n > best[0]:
                        best = (n, x, y)
        if best is None:
            rejected.append((c["pdb_id"], "no chain pair with 50-600 Cα residues")); continue
        if best[0] < 10:
            rejected.append((c["pdb_id"], f"only {best[0]} Cα contacts")); continue
        chosen.append((c["pdb_id"], best[1], best[2], "train"))
    chosen += [(p, x, y, "heldout") for p, x, y in HELDOUT]

    n_res_pairs = n_contacts = 0
    for pid, x, y, split in chosen:
        chs = chains_ca(fetch(pid))
        write_chain(pid, x, chs[x]); write_chain(pid, y, chs[y])
        D = dist(chs[x], chs[y])
        np.savez(Path("data/distances") / f"{pid.lower()}_{x}{y}.npz", D=D,
                 res_a=np.array([r[0] for r in chs[x]]), res_b=np.array([r[0] for r in chs[y]]))
        if split == "train":
            n_res_pairs += D.size
            n_contacts += int((D < 7).sum())
    with open("data/pairs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pdb_id", "chain_a", "chain_b", "split"]); w.writerows(chosen)
    with open("data/substitutions.csv", "w", newline="") as f:
        csv.writer(f).writerow(["requested_id", "used_id", "reason"])
    with open("data/curation/rejected.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pdb_id", "reason"]); w.writerows(rejected)
    n_train = sum(1 for c in chosen if c[3] == "train")
    summary = {"n_pairs": n_train, "n_residue_pairs": n_res_pairs, "n_contacts": n_contacts,
               "note": "Substitute training set curated from RCSB (paper's 464-pair list unavailable); see scripts/n3_curate.py"}
    Path("data/dataset_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), f"rejected {len(rejected)}")


if __name__ == "__main__":
    main()
