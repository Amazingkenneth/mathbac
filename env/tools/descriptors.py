"""Per-chain graph descriptors computed with the released code's own functions.

    python descriptors.py gnm       --pdb X.pdb --chain A --out gnm.csv [--n-modes 10]
    python descriptors.py curvature --ca data/chains/x_A.csv --out curv.csv [--cutoff 7]
    python descriptors.py mfd       --ca data/chains/x_A.csv --out mfd.csv [--cutoff 7] [--box-radius 5]
    python descriptors.py opd       --ca data/chains/x_A.csv --out opd.csv

--ca files follow the artifact contract (resnum,resname,x,y,z). Output rows follow the same order.
Runs inside the sandbox image (Python 3.8 with the capsule's pinned packages).
"""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

sys.path.insert(0, "/opt/unified_fast/code/unified_descriptors")


def ca_graph(ca, cutoff):
    import networkx as nx

    xyz = ca[["x", "y", "z"]].to_numpy(float)
    d = cdist(xyz, xyz)
    G = nx.Graph()
    G.add_nodes_from(range(len(xyz)))
    i, j = np.where((d <= cutoff) & (d > 0))
    G.add_edges_from((int(a), int(b)) for a, b in zip(i, j) if a < b)
    return G


def write(ca, cols, out):
    df = pd.DataFrame({"resnum": ca["resnum"], "resname": ca["resname"]})
    for k, v in cols.items():
        df[k] = np.asarray(v)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["gnm", "curvature", "mfd", "opd"])
    p.add_argument("--pdb")
    p.add_argument("--chain")
    p.add_argument("--ca")
    p.add_argument("--out", required=True)
    p.add_argument("--cutoff", type=float, default=7.0)
    p.add_argument("--n-modes", type=int, default=10)
    p.add_argument("--box-radius", type=int, default=5)
    a = p.parse_args()

    if a.kind == "gnm":
        import GNM

        vals = GNM.gnm_sum_mode(a.pdb, a.n_modes, a.chain)
        pd.DataFrame({"index": range(len(vals)), "gnm": vals}).to_csv(a.out, index=False)
        print(f"wrote {len(vals)} GNM values to {a.out} (align rows with data/chains before merging)")
        return

    ca = pd.read_csv(a.ca, dtype={"resnum": str})
    if a.kind == "curvature":
        import GT_curvature

        G = ca_graph(ca, a.cutoff)
        write(ca, {"orc": GT_curvature.ollivier_ricci(G), "frc": GT_curvature.forman_ricci(G)}, a.out)
    elif a.kind == "mfd":
        import MFD

        G = ca_graph(ca, a.cutoff)
        fd = MFD.fractal_dimension(G)
        more = MFD.more_box(G, a.box_radius)
        write(ca, {"mfd": fd[:, 1], "mfd_1": more[:, 1], "mfd_2": more[:, 2], "mfd_3": more[:, 3], "mfd_4": more[:, 4]}, a.out)
    elif a.kind == "opd":
        import local_OPD

        # local_OPD reads biopandas-style columns; residue_name selects the residue mass used as the OPD weight.
        c1 = ca.rename(columns={"x": "x_coord", "y": "y_coord", "z": "z_coord", "resname": "residue_name"})
        arr = local_OPD.GOS_single(c1)
        write(ca, {"opd_5": arr[:, 0], "opd_7": arr[:, 1], "opd_10": arr[:, 2], "opd_15": arr[:, 3]}, a.out)


if __name__ == "__main__":
    main()
