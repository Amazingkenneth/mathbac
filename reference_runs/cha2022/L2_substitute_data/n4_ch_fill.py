"""N4a fix-up: residues PDB2PQR dropped (0.09%: 79 residues of 1EUV chain B) get the per-residue-type mean
of this run's own PDB2PQR values for every charge/count column. No external tables are used."""
import csv, glob, collections, math
cols = ["c_charge", "n_charge", "o_charge", "h_charge", "s_charge", "n_count", "c_count", "o_count", "h_count", "s_count"]
acc = collections.defaultdict(lambda: collections.defaultdict(list))
files = sorted(glob.glob("features/CH/*.csv"))
for f in files:
    for r in csv.DictReader(open(f)):
        for c in cols:
            if r[c] not in ("nan", ""):
                acc[r["resname"]][c].append(float(r[c]))
mean = {n: {c: sum(v) / len(v) for c, v in d.items()} for n, d in acc.items()}
fixed = 0
for f in files:
    rows = list(csv.DictReader(open(f)))
    if not any(r["c_charge"] in ("nan", "") for r in rows):
        continue
    for r in rows:
        if r["c_charge"] in ("nan", ""):
            for c in cols:
                r[c] = round(mean.get(r["resname"], {}).get(c, 0.0), 4)
            fixed += 1
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("filled", fixed, "residues")
