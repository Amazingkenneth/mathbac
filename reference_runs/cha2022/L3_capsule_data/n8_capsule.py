"""N3c/N4d/N8 (L3): protein-nanoparticle predictions with the GE_GT DNN trained in N6b on the authors' arrays.

Features: the authors' precomputed protein-NP arrays (capsule processed_test_data/PNI) for hMPO-SWNT, the two hIAPP
cases and the protein-corona set. PSM-α has no precomputed array, so its features are computed here with the
stage-N4 pipeline (GHECOM approximation for rinacc/shell/pocketness) on the authors' published 5I55 and GQD (G3)
structures from the Fig. 5 source data; marked as a hybrid in the case list.

Rows are protein-residue major, NP-node minor. Interface = top 0.1% of pairs. Corona score (not defined in the
released code): fraction of residue-node pairs with predicted contact probability >= 0.5.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from n7_capsule import dnn, features  # noqa: E402

CAP = Path("materials/precomputed")
TOP = 0.001
PROTEIN = {"apolipoprotein": "3k2s", "histidine-rich glycoprotein": "4ccv", "galectin-3-binding protein": "6gfb", "albumin": "6hsc"}
NPS = {"C_5nm_sphere_reduced_13": "spherical_C_5nm", "SWNT_62_50nm_5A_semi_reduced": "SWNT_62_50nm_5A",
       "SWNT_66_50nm_8A_metal_reduced": "SWNT_66_50nm_8A", "SWNT_78_50nm_1nm_reduced": "SWNT_78_50nm_1nm",
       "SWNT_87_50nm_1nm_reduced": "SWNT_87_50nm_1nm"}
CASES = [  # case, array file, protein file, chain, np name, reference
    ("CNT-1CXP", "data_feature_1cxp_AC_ASWNT_14_0_500A_13_reduced.npy", "1cxp_AC", "A", "SWNT_14_0_500A_13_reduced", "Fig. 5d,e"),
    ("GQD-2L86", "data_feature_2l86_Aiapp_gqd_atomic.npy", "2l86", "A", "iapp_gqd_atomic", "Fig. 5b"),
    ("GQD-6ZRF", "data_feature_6zrf_combined_Aiapp_gqd_atomic.npy", "6zrf_combined", "A", "iapp_gqd_atomic", "Fig. 5c"),
]


def residues(fname, chain):
    seen, out = set(), []
    for line in open(CAP / "pdb_protein" / f"{fname}.pdb", errors="replace"):
        if line.startswith("ENDMDL"):
            break
        if line.startswith("ATOM") and line[21] == chain and line[12:16].strip() == "CA" and line[16] in " A":
            rn = str(int(line[22:26]))
            if rn not in seen:
                seen.add(rn); out.append(rn)
    return out


def interface(s, res, n_nodes):
    ip, jn = np.divmod(np.arange(len(s)), n_nodes)
    n = max(int(len(s) * TOP), 1)
    top = np.argsort(s)[-n:]
    return {"protein_residues": sorted({res[ip[k]] for k in top}, key=lambda x: (len(x), x)),
            "np_nodes": sorted({int(jn[k]) + 1 for k in top}), "top_fraction": TOP, "n_pairs_scored": int(len(s))}


def save_case(case, s, res, n_nodes, extra):
    Path("predictions/pni").mkdir(parents=True, exist_ok=True)
    ip, jn = np.divmod(np.arange(len(s)), n_nodes)
    with open(f"predictions/pni/{case}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["np_node", "res_protein", "prob"])
        for a, b, p in zip(jn, ip, s):
            w.writerow([a + 1, res[b], f"{p:.6f}"])
    iface = {**interface(s, res, n_nodes), **extra}
    Path(f"predictions/pni/{case}_interface.json").write_text(json.dumps(iface))
    print(case, len(iface["protein_residues"]), "residues:", iface["protein_residues"][:30], flush=True)


def psm_alpha():
    """Hybrid: features computed with the N4 pipeline (GHECOM approximation) for 5I55 + GQD G3."""
    sys.path.insert(0, "/srv/mathbac/runs/opus-direct-L2/workspace/scripts")
    import n8_pni as L2

    L2.NPS = Path("/srv/mathbac/runs/opus-direct-L2/workspace/materials/np_structures")
    xyz, el = L2.np_atoms("Fig5a_G3.pdb")
    NF = L2.graph_features(xyz, ["MOL"] * len(xyz), 4.0, 4.0, xyz, np.arange(len(xyz)))
    ca, atoms = L2.protein_residues("5I55", ["A"], None)
    keys = [k for k, _, _ in ca]
    idx = {k: i for i, k in enumerate(keys)}
    pxyz = np.array([p for _, _, p in ca])
    axyz = np.array([p for k, p in atoms if k in idx]); aown = np.array([idx[k] for k, _ in atoms if k in idx])
    PF = L2.graph_features(pxyz, [r for _, r, _ in ca], 7.0, 7.0, axyz, aown)
    ip, jn = np.meshgrid(np.arange(len(PF)), np.arange(len(NF)), indexing="ij")
    X = np.hstack([PF[ip.ravel()], NF[jn.ravel()]])
    return dnn("GE_GT", X), [k[1] for k in keys], len(NF)


def main():
    Path("data/np").mkdir(parents=True, exist_ok=True)
    Path("features/np").mkdir(parents=True, exist_ok=True)
    cases_rows = []
    for case, arr, pfile, ch, npn, ref in CASES:
        T = np.load(CAP / "processed_test_data/PNI" / arr, allow_pickle=True).astype(float)
        res = residues(pfile, ch)
        n_nodes = len(T) // len(res)
        assert n_nodes * len(res) == len(T), (case, len(T), len(res))
        save_case(case, dnn("GE_GT", features(T, "GE_GT")), res, n_nodes, {"features": f"capsule {arr}"})
        cases_rows.append([case, npn, pfile.split("_")[0], "AC" if "1cxp" in pfile else ch, ref])
    s, res, n_nodes = psm_alpha()
    save_case("GQD-PSMa", s, res, n_nodes, {"features": "hybrid: N4 pipeline with GHECOM approximation on 5I55 + G3"})
    cases_rows.append(["GQD-PSMa", "gqd_g3", "5i55", "A", "Fig. 5a (hybrid features)"])
    rank = []
    for prot, pdb in PROTEIN.items():
        for npfile, npname in NPS.items():
            tag = "AC_" if npfile.startswith("C_") else "A"
            T = np.load(CAP / f"processed_test_data/PNI/data_feature_{pdb}_{tag}{npfile.replace('C_5nm', '5nm') if tag == 'AC_' else npfile}.npy",
                        allow_pickle=True).astype(float)
            p = dnn("GE_GT", features(T, "GE_GT"))
            rank.append([npname, prot, round(float((p >= 0.5).mean()), 4)])
            print("corona", npname, prot, rank[-1][2], flush=True)
    with open("predictions/pni/corona_ranking.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["np_name", "protein", "score"]); w.writerows(rank)
    for npn in sorted({r[1] for r in cases_rows}):
        src = CAP / "pdb_np" / f"{npn}.pdb"
        if not src.exists():
            src = Path("/srv/mathbac/runs/opus-direct-L2/workspace/materials/np_structures/Fig5a_G3.pdb") if npn == "gqd_g3" else None
        with open(f"data/np/{npn}_cg.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["node", "x", "y", "z", "n_atoms", "element"])
            if src is not None:
                k = 0
                for line in open(src):
                    if line.startswith(("ATOM", "HETATM")):
                        k += 1
                        w.writerow([k, line[30:38].strip(), line[38:46].strip(), line[46:54].strip(),
                                    13 if "13_reduced" in npn else 1, (line[76:78].strip() or "C")])
            else:  # coarse-grained file is not in the capsule; nodes are known only through the precomputed features
                for i in range(n_nodes if npn == "SWNT_14_0_500A_13_reduced" else 1):
                    w.writerow([i + 1, "", "", "", 13, "C"])
    with open("data/np/cases.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case", "np_name", "pdb_id", "chain", "reference"]); w.writerows(cases_rows)


if __name__ == "__main__":
    main()
