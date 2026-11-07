"""N4a-c: per-residue CH, GE and GT descriptors for every chain in data/pairs.csv.

Run with the capsule environment (Python 3.8):
    /opt/mathbac/env/bin/python scripts/n4_descriptors.py --stage fast   # OPD, MFD, GNM, GE-approx (process pool)
    /opt/mathbac/env/bin/python scripts/n4_descriptors.py --stage orc    # Ollivier/Forman-Ricci (ORC runs its own pool)
    python3 scripts/n4_ch.py                                             # CH via PDB2PQR (separate script)
    /opt/mathbac/env/bin/python scripts/n4_descriptors.py --stage write  # merge into features/GE and features/GT

GT and OPD use the released code's own functions (GT_curvature, MFD, local_OPD; GNM with the released
settings: Kirchhoff cutoff 7 Å, first 10 modes, eigenvectors rounded to 3 decimals).

GHECOM is not available (source download needs registration), so rinacc / shell / pocketness are an
APPROXIMATION computed here by multiscale mathematical morphology on a 1 Å grid, in the spirit of
GHECOM (Kawabata 2010): a grid point outside the protein is "inaccessible" to probe radius R if no
probe sphere of radius R that avoids the protein's atoms covers it. For each residue, over grid points
in a 3 Å shell outside its atoms' van der Waals surface:
  rinacc     = mean over shell points of the smallest probe radius (2-10 Å) that cannot reach the point (10 if all can)
  shell      = volume (Å^3) of shell points reachable by a 1.4 Å water probe
  pocketness = mean over shell points of (10 - rinacc), i.e. how deep inside a pocket the residue's surroundings are
"""

import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, "/opt/unified_fast/code/unified_descriptors")

CACHE = Path("features/_parts")
VDW = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "H": 1.1}
RADII = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


def chains():
    out = []
    for r in csv.DictReader(open("data/pairs.csv")):
        for ch in (r["chain_a"], r["chain_b"]):
            k = (r["pdb_id"].lower(), ch)
            if k not in out:
                out.append(k)
    return out


def ca_table(pdb, ch):
    import pandas as pd

    return pd.read_csv(f"data/chains/{pdb}_{ch}.csv", dtype={"resnum": str})


def ca_graph(xyz, cutoff=7.0):
    import networkx as nx
    from scipy.spatial.distance import cdist

    d = cdist(xyz, xyz)
    G = nx.Graph()
    G.add_nodes_from(range(len(xyz)))
    i, j = np.where((d <= cutoff) & (d > 0))
    G.add_edges_from((int(a), int(b)) for a, b in zip(i, j) if a < b)
    return G


def heavy_atoms(pdb, ch, resnums):
    """Heavy atoms of the chain's Cα residues (ATOM, first model, altloc blank/A), with residue index."""
    idx = {r: k for k, r in enumerate(resnums)}
    xyz, rad, owner = [], [], []
    for line in open(f"data/pdb/{pdb.upper()}.pdb", errors="replace"):
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM") or line[21] != ch or line[16] not in " A":
            continue
        el = (line[76:78].strip() or line[12:16].strip()[0]).upper()
        if el == "H":
            continue
        rn = line[22:27].strip()
        if rn not in idx:
            continue
        xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        rad.append(VDW.get(el[:1], 1.7))
        owner.append(idx[rn])
    return np.array(xyz), np.array(rad), np.array(owner)


def ghecom_like(xyz, rad, owner, n_res, spacing=1.0, shell_thick=3.0):
    from scipy import ndimage

    pad = max(RADII) + shell_thick + 3
    lo = xyz.min(0) - pad
    shape = np.ceil((xyz.max(0) + pad - lo) / spacing).astype(int) + 1
    grid_idx = np.round((xyz - lo) / spacing).astype(int)
    # distance from each grid point to the nearest atom surface: EDT to atom centres minus radius is approximated
    # by marking atom centres and subtracting the mean radius (1.7 Å) - adequate on a 1 Å grid.
    centres = np.ones(shape, bool)
    centres[tuple(grid_idx.T)] = False
    d_centre = ndimage.distance_transform_edt(centres, sampling=spacing)
    d_surf = d_centre - 1.7
    protein = d_surf <= 0
    rinacc = np.full(shape, max(RADII), float)
    for R in sorted(RADII, reverse=True):
        allowed = d_surf >= R                       # probe centres that keep a radius-R sphere clear of atoms
        reach = ndimage.distance_transform_edt(~allowed, sampling=spacing) <= R  # points covered by some such sphere
        rinacc[~reach & ~protein] = R               # smaller R overwrite: the smallest radius that cannot reach
    water = ndimage.distance_transform_edt(~(d_surf >= 1.4), sampling=spacing) <= 1.4
    # assign shell points (outside protein, within shell_thick of the surface) to the nearest residue
    shell_mask = (~protein) & (d_surf <= shell_thick)
    pts = np.argwhere(shell_mask)
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)
    _, nearest = tree.query(pts * spacing + lo)
    res_of = owner[nearest]
    r_vals = rinacc[tuple(pts.T)]
    w_vals = water[tuple(pts.T)]
    out = np.zeros((n_res, 3))
    for k in range(n_res):
        m = res_of == k
        if m.any():
            out[k] = (r_vals[m].mean(), w_vals[m].sum() * spacing ** 3, (max(RADII) - r_vals[m]).mean())
        else:  # buried residue: no shell points
            out[k] = (0.0, 0.0, max(RADII))
    return out


def fast(job):
    pdb, ch = job
    part = CACHE / f"{pdb}_{ch}.fast.json"
    if part.exists():
        return job, "cached"
    import local_OPD
    import MFD
    import prody

    prody.confProDy(verbosity="none")
    ca = ca_table(pdb, ch)
    xyz = ca[["x", "y", "z"]].to_numpy(float)
    G = ca_graph(xyz)
    fd = MFD.fractal_dimension(G)
    more = MFD.more_box(G, 5)
    # fractal_dimension / more_box return rows keyed by node id; align to node order and keep NaN for nodes left out
    def by_node(arr, col):
        m = {int(r[0]): r[col] for r in arr}
        return [float(m.get(i, np.nan)) for i in range(len(xyz))]
    c1 = ca.rename(columns={"x": "x_coord", "y": "y_coord", "z": "z_coord", "resname": "residue_name"})
    n = len(xyz)
    if n >= 15:
        opd = local_OPD.GOS_single(c1)
    else:
        opd = np.full((n, 4), np.nan)
    gnm = prody.GNM("chain")
    gnm.buildKirchhoff(xyz, cutoff=7.0)
    gnm.calcModes("all")
    k = min(10, gnm.numModes())
    gsum = np.sum(np.square(gnm[0:k].getEigvecs().round(3)), axis=1)
    hx, hr, ho = heavy_atoms(pdb, ch, list(ca["resnum"]))
    ge = ghecom_like(hx, hr, ho, n)
    res = {"mfd": by_node(fd, 1), **{f"mfd_{j}": by_node(more, j) for j in range(1, 5)},
           "opd_5": opd[:, 0].tolist(), "opd_7": opd[:, 1].tolist(), "opd_10": opd[:, 2].tolist(), "opd_15": opd[:, 3].tolist(),
           "gnm": gsum.tolist(), "rinacc": ge[:, 0].tolist(), "shell": ge[:, 1].tolist(), "pocketness": ge[:, 2].tolist()}
    part.write_text(json.dumps(res))
    return job, "done"


def orc_stage(jobs):
    import GT_curvature

    for pdb, ch in jobs:
        part = CACHE / f"{pdb}_{ch}.orc.json"
        if part.exists():
            continue
        xyz = ca_table(pdb, ch)[["x", "y", "z"]].to_numpy(float)
        G = ca_graph(xyz)
        res = {"orc": [float(v) for v in GT_curvature.ollivier_ricci(G)], "frc": [float(v) for v in GT_curvature.forman_ricci(G)]}
        part.write_text(json.dumps(res))
        print("orc", pdb, ch, flush=True)


def write_stage(jobs):
    import pandas as pd

    for pdb, ch in jobs:
        ca = ca_table(pdb, ch)
        f = json.loads((CACHE / f"{pdb}_{ch}.fast.json").read_text())
        o = json.loads((CACHE / f"{pdb}_{ch}.orc.json").read_text())
        base = ca[["resnum", "resname"]].copy()
        ge = base.copy()
        for c in ["rinacc", "shell", "pocketness", "opd_5", "opd_7", "opd_10", "opd_15"]:
            ge[c] = f[c]
        gt = base.copy()
        gt["orc"], gt["frc"], gt["gnm"] = o["orc"], o["frc"], f["gnm"]
        for c in ["mfd", "mfd_1", "mfd_2", "mfd_3", "mfd_4"]:
            gt[c] = f[c]
        # residues the released routines leave undefined (small components, chains < 15 residues for OPD):
        # forward/back fill along the chain, as the released prediction code does (fillna(method='ffill'))
        ge = ge.fillna(method="ffill").fillna(method="bfill")
        gt = gt.fillna(method="ffill").fillna(method="bfill")
        Path("features/GE").mkdir(parents=True, exist_ok=True)
        Path("features/GT").mkdir(parents=True, exist_ok=True)
        ge.to_csv(f"features/GE/{pdb}_{ch}.csv", index=False)
        gt.to_csv(f"features/GT/{pdb}_{ch}.csv", index=False)
    print(f"wrote GE and GT for {len(jobs)} chains")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fast", "orc", "write"], required=True)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = chains()
    if a.stage == "fast":
        with Pool(a.procs) as pool:
            for k, (job, state) in enumerate(pool.imap_unordered(fast, jobs), 1):
                if k % 25 == 0 or k == len(jobs):
                    print(f"fast {k}/{len(jobs)}", flush=True)
    elif a.stage == "orc":
        orc_stage(jobs)
    else:
        write_stage(jobs)


if __name__ == "__main__":
    main()
