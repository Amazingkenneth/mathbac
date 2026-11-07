"""N3 (L3): the authors' pair list (capsule data/PPI_list/PPI_data.csv) and structures (data/pdb_protein).

Residues: ATOM records with a Cα, first model, altloc blank or A, as the released code reads them.
Duplicate rows in the list are kept once; pairs whose structure file is missing are recorded in
data/substitutions.csv. Held-out complexes 6WZO, 6NQN, 1C29 (chains A, B) are added as the heldout split.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from n3_build import chains_ca, dist, write_chain  # noqa: E402

CAP = Path("materials/precomputed")
HELDOUT = [("6WZO", "A", "B"), ("6NQN", "A", "B"), ("1C29", "A", "B")]


def main():
    for d in ("data/chains", "data/distances"):
        Path(d).mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(CAP / "PPI_list/PPI_data.csv")))
    seen, pairs, missing = set(), [], []
    for r in rows:
        k = (r["PDB ID"].strip().upper(), r["chain1"].strip(), r["chain2"].strip())
        if k in seen:
            continue
        seen.add(k)
        f = CAP / "pdb_protein" / f"{k[0].lower()}.pdb"
        if not f.exists():
            missing.append((k[0], "", "structure file not in capsule"))
            continue
        pairs.append((*k, "train", f))
    held = []
    for pid, a, b in HELDOUT:
        f = CAP / "pdb_protein" / f"{pid.lower()}.pdb"
        if not f.exists():
            f = Path(f"data/pdb/{pid}.pdb")
            f.parent.mkdir(exist_ok=True, parents=True)
            if not f.exists():
                import urllib.request
                urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pid}.pdb", f)
        held.append((pid, a, b, "heldout", f))
    out, n_res, n_con, dropped = [], 0, 0, []
    for pid, a, b, split, f in pairs + held:
        chs = chains_ca(f)
        if a not in chs or b not in chs or len(chs[a]) < 5 or len(chs[b]) < 5:
            dropped.append((pid, a, b))
            missing.append((pid, "", f"chain {a} or {b} has no Cα records"))
            continue
        write_chain(pid, a, chs[a]); write_chain(pid, b, chs[b])
        D = dist(chs[a], chs[b])
        np.savez(Path("data/distances") / f"{pid.lower()}_{a}{b}.npz", D=D,
                 res_a=np.array([r[0] for r in chs[a]]), res_b=np.array([r[0] for r in chs[b]]))
        if split == "train":
            n_res += D.size
            n_con += int((D < 7).sum())
        out.append([pid, a, b, split])
    with open("data/pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["pdb_id", "chain_a", "chain_b", "split"]); w.writerows(out)
    with open("data/substitutions.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["requested_id", "used_id", "reason"]); w.writerows(missing)
    n_train = sum(1 for r in out if r[3] == "train")
    summary = {"n_pairs": n_train, "n_residue_pairs": n_res, "n_contacts": n_con,
               "note": f"authors' list: {len(rows)} rows, {len(seen)} unique pairs; paper reports 464 pairs and 27,859,297 residue pairs"}
    Path("data/dataset_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), "dropped", dropped)


if __name__ == "__main__":
    main()
