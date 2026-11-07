"""N3c + N4d + N8: protein-nanoparticle cases.

Follows the released PNI_descriptors_cal.py rather than the paper text where they differ:
- NP nodes are the atoms of the NP model (the released code reads every ATOM record of the NP file as a node,
  and the published Fig. 5 NP indices are atom indices), NP graph edges at <= 4 Å, GNM on NP atoms at 4 Å.
  The paper text instead describes 13-atom coarse-grained nodes with 7 Å edges.
- Protein side: Cα residues with the same GE + GT descriptors as the training chains.
Inputs: NP models and the 6ZRF combined fibril from the paper's Fig. 5 source data (materials/np_structures);
protein structures from RCSB. Pair features = [protein GE+GT | NP GE+GT], scored by the GE_GT DNN fold
ensemble from N6b; the top 0.1% of pairs give the predicted sites.

Not attempted (no published models to build them from): ZnO nanopyramid-β-galactosidase, and the protein-corona
ranking (5 nm carbon sphere and three 50 nm SWNTs with albumin, apolipoprotein, HRG and galectin-3BP).

    /opt/mathbac/env/bin/python scripts/n8_pni.py
"""

import csv
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "/opt/unified_fast/code/unified_descriptors")
from n4_descriptors import RADII, ca_graph, ghecom_like  # noqa: E402

NPS = Path("materials/np_structures")
TOP = 0.001
CASES = [
    # case, NP name, NP file, protein PDB id, chains, protein file (None = RCSB), reference
    ("CNT-1CXP", "swnt_14_0_17A", "Fig5de_SWNT_14_0_17A.pdb", "1CXP", ["A", "C"], None, "Fig. 5d,e; Kagan et al."),
    ("GQD-PSMa", "gqd_g3", "Fig5a_G3.pdb", "5I55", ["A"], None, "Fig. 5a (text: 5KHB)"),
    ("GQD-2L86", "gqd_hydroxylated", "Fig5bc_iapp_gqd_atomic.pdb", "2L86", ["A"], None, "Fig. 5b; Faridi et al., Wang et al."),
    ("GQD-6ZRF", "gqd_hydroxylated", "Fig5bc_iapp_gqd_atomic.pdb", "6ZRF", ["A"], "Fig5c_6zrf_combined.pdb", "Fig. 5c"),
]
GE_COLS = ["rinacc", "shell", "pocketness", "opd_5", "opd_7", "opd_10", "opd_15"]
GT_COLS = ["orc", "frc", "gnm", "mfd", "mfd_1", "mfd_2", "mfd_3", "mfd_4"]


def protein_residues(pid, chains, local):
    path = NPS / local if local else Path(f"data/pdb/{pid}.pdb")
    if not path.exists():
        with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.pdb", timeout=60) as r:
            path.write_bytes(r.read())
    ca, atoms, seen = [], [], set()
    for line in open(path, errors="replace"):
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM") or line[21] not in chains or line[16] not in " A":
            continue
        key = (line[21], line[22:27].strip())
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        el = (line[76:78].strip() or line[12:16].strip()[0]).upper()
        if line[12:16].strip() == "CA" and key not in seen:
            seen.add(key)
            ca.append((key, line[17:20].strip(), xyz))
        if el != "H":
            atoms.append((key, xyz))
    return ca, atoms


def np_atoms(fname):
    rows = [l for l in open(NPS / fname) if l.startswith(("ATOM", "HETATM"))]
    xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in rows])
    el = [(l[76:78].strip() or l[12:16].strip()[0]).upper() for l in rows]
    return xyz, el


def graph_features(xyz, names, cutoff, gnm_cutoff, atom_radius_xyz, atom_owner):
    import GT_curvature
    import local_OPD
    import MFD
    import pandas as pd
    import prody

    prody.confProDy(verbosity="none")
    n = len(xyz)
    G = ca_graph(xyz, cutoff)
    orc = GT_curvature.ollivier_ricci(G)
    frc = GT_curvature.forman_ricci(G)
    fd = {int(r[0]): r[1] for r in MFD.fractal_dimension(G)}
    more = {int(r[0]): r[1:5] for r in MFD.more_box(G, 5)}
    g = prody.GNM("x")
    g.buildKirchhoff(xyz, cutoff=gnm_cutoff)
    g.calcModes("all")
    gnm = np.sum(np.square(g[0: min(10, g.numModes())].getEigvecs().round(3)), axis=1)
    c1 = pd.DataFrame({"x_coord": xyz[:, 0], "y_coord": xyz[:, 1], "z_coord": xyz[:, 2], "residue_name": names})
    opd = local_OPD.GOS_single(c1)
    ge = ghecom_like(atom_radius_xyz, np.full(len(atom_radius_xyz), 1.7), atom_owner, n)
    rows = []
    for i in range(n):
        m = more.get(i, [np.nan] * 4)
        rows.append([ge[i, 0], ge[i, 1], ge[i, 2], *opd[i], orc[i], frc[i], gnm[i], fd.get(i, np.nan), *m])
    import pandas as pd

    df = pd.DataFrame(rows, columns=GE_COLS + GT_COLS).fillna(method="ffill").fillna(method="bfill")
    return df.to_numpy(float)


def dnn_scores(X):
    from tensorflow import keras

    d = Path("ml/models/dnn_GE_GT")
    sc = np.load(d / "scaler.npz")
    rng = np.where(sc["data_max"] > sc["data_min"], sc["data_max"] - sc["data_min"], 1.0)
    Xs = (X - sc["data_min"]) / rng
    return np.mean([keras.models.load_model(str(f), compile=False).predict(Xs, batch_size=4096)[:, 1]
                    for f in sorted(d.glob("fold*.h5"))], axis=0)


def main():
    for d in ("data/np", "features/np", "predictions/pni"):
        Path(d).mkdir(parents=True, exist_ok=True)
    with open("data/np/cases.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "np_name", "pdb_id", "chain", "reference"])
        for case, npn, _, pid, chains, _, ref in CASES:
            w.writerow([case, npn, pid.lower(), "".join(chains), ref])
    np_cache = {}
    for case, npn, nfile, pid, chains, local, ref in CASES:
        if npn not in np_cache:
            xyz, el = np_atoms(nfile)
            with open(f"data/np/{npn}_cg.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["node", "x", "y", "z", "n_atoms", "element"])
                for i, (p, e) in enumerate(zip(xyz, el)):
                    w.writerow([i + 1, *np.round(p, 3), 1, e])
            F = graph_features(xyz, ["MOL"] * len(xyz), 4.0, 4.0, xyz, np.arange(len(xyz)))
            with open(f"features/np/{npn}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["node", *GE_COLS, *GT_COLS])
                for i, r in enumerate(F):
                    w.writerow([i + 1, *np.round(r, 6)])
            np_cache[npn] = F
        NF = np_cache[npn]
        ca, atoms = protein_residues(pid, chains, local)
        keys = [k for k, _, _ in ca]
        idx = {k: i for i, k in enumerate(keys)}
        pxyz = np.array([p for _, _, p in ca])
        axyz = np.array([p for _, p in atoms if _ in idx])
        aown = np.array([idx[k] for k, _ in atoms if k in idx])
        PF = graph_features(pxyz, [r for _, r, _ in ca], 7.0, 7.0, axyz, aown)
        ip, jn = np.meshgrid(np.arange(len(PF)), np.arange(len(NF)), indexing="ij")
        ip, jn = ip.ravel(), jn.ravel()
        s = dnn_scores(np.hstack([PF[ip], NF[jn]]))
        with open(f"predictions/pni/{case}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["np_node", "res_protein", "prob"])
            for a, b, p in zip(jn, ip, s):
                w.writerow([a + 1, keys[b][1], f"{p:.6f}"])
        n = max(int(len(s) * TOP), 1)
        top = np.argsort(s)[-n:]
        res = sorted({keys[ip[k]][1] for k in top}, key=lambda x: (len(x), x))
        out = {"protein_residues": res, "protein_chains": chains, "np_nodes": sorted({int(jn[k]) + 1 for k in top}),
               "top_fraction": TOP, "n_pairs_scored": int(len(s))}
        Path(f"predictions/pni/{case}_interface.json").write_text(json.dumps(out))
        print(case, len(res), "protein residues:", res[:25], flush=True)
    with open("predictions/pni/corona_ranking.csv", "w", newline="") as f:
        csv.writer(f).writerow(["np_name", "protein", "score"])  # not attempted: no published NP/protein models


if __name__ == "__main__":
    main()
