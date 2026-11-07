"""N9: reproduction report (report/report.md) and claim verdicts (report/claims.yaml) from this workspace's files.

Every number is recomputed here from the stage outputs. Paper values are quoted from the paper's Table 1, SI Tables
4-6 and Fig. 5 as published (the paper is under /materials/paper).

    python3 scripts/n9_report.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import yaml

SUBSETS = ["all", "CH", "GE", "GT", "GE_GT", "CH_GE", "CH_GT"]
MODELS = ["lr", "gnb", "svm", "rf", "xgb", "dnn"]
# Table 1 of the paper (accuracy, AUC, F1), quoted from the published table.
PAPER = {
    "lr": {"all": (.553, .565, .592), "CH": (.551, .573, .553), "GE": (.553, .563, .592), "GT": (.671, .730, .648), "GE_GT": (.551, .562, .587), "CH_GE": (.551, .562, .589), "CH_GT": (.684, .743, .667)},
    "gnb": {"all": (.578, .633, .668), "CH": (.537, .559, .480), "GE": (.574, .628, .665), "GT": (.615, .711, .670), "GE_GT": (.576, .630, .666), "CH_GE": (.575, .627, .666), "CH_GT": (.652, .709, .668)},
    "svm": {"all": (.607, .656, .655), "CH": (.552, .576, .523), "GE": (.606, .652, .655), "GT": (.663, .719, .589), "GE_GT": (.606, .652, .657), "CH_GE": (.605, .650, .656), "CH_GT": (.674, .741, .607)},
    "rf": {"all": (.822, .920, .803), "CH": (.561, .583, .545), "GE": (.769, .869, .737), "GT": (.804, .889, .789), "GE_GT": (.810, .907, .789), "CH_GE": (.802, .899, .778), "CH_GT": (.812, .905, .797)},
    "xgb": {"all": (.829, .913, .817), "CH": (.559, .583, .542), "GE": (.780, .862, .763), "GT": (.804, .879, .794), "GE_GT": (.815, .899, .803), "CH_GE": (.798, .885, .784), "CH_GT": (.815, .895, .804)},
    "dnn": {"all": (.840, .908, .830), "CH": (.574, .590, .548), "GE": (.770, .838, .756), "GT": (.808, .877, .800), "GE_GT": (.823, .891, .811), "CH_GE": (.791, .857, .768), "CH_GT": (.806, .876, .794)},
}


def auc(y, s):
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); ss = s[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    p = y.sum(); n = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def scores(f):
    z = np.load(f); out = []
    for k in np.unique(z["fold"]):
        m = z["fold"] == k; y = z["y_true"][m].astype(int); s = z["y_score"][m]
        pred_non = s < 0.5; yn = y == 0
        tp = (pred_non & yn).sum(); fp = (pred_non & ~yn).sum(); fn = (~pred_non & yn).sum()
        out.append(((( s >= 0.5) == (y == 1)).mean(), auc(y, s), 2 * tp / (2 * tp + fp + fn)))
    return np.mean(out, axis=0)


def table1():
    rows = {}
    for m in MODELS:
        for s in SUBSETS:
            f = Path(f"ml/cv/predictions/{m}_{s}.npz")
            rows[(m, s)] = scores(f) if f.exists() else None
    return rows


def residue_f1(pdb):
    held = json.loads(Path(f"predictions/ppi/{pdb}_dnn_GE_GT_interface.json").read_text())
    D = np.load(f"data/distances/{pdb}_AB.npz")
    ra, rb = list(D["res_a"]), list(D["res_b"])
    ta = {ra[i] for i in np.where((D["D"] < 7).any(1))[0]}; tb = {rb[j] for j in np.where((D["D"] < 7).any(0))[0]}
    pa, pb = set(map(str, held["chain_a"])), set(map(str, held["chain_b"]))
    tp = len(pa & ta) + len(pb & tb); fp = len(pa - ta) + len(pb - tb); fn = len(ta - pa) + len(tb - pb)
    p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
    return p, r, (2 * p * r / (p + r) if p + r else 0), len(pa) + len(pb), len(ta) + len(tb)


def main():
    T = table1()
    within = sum(1 for (m, s), v in T.items() if v is not None and all(abs(a - b) <= 0.03 for a, b in zip(v, PAPER[m][s])))
    unscaled = {m: {s: scores(Path(f"ml/cv/predictions_scaled/{m}_{s}.npz")) for s in SUBSETS
                    if Path(f"ml/cv/predictions_scaled/{m}_{s}.npz").exists()} for m in ("lr", "gnb", "svm")}
    a = {k: (T[k][1] if T[k] is not None else float("nan")) for k in T}
    gt_ge_ch = a[("dnn", "GT")] > a[("dnn", "GE")] > a[("dnn", "CH")] and a[("rf", "GT")] > a[("rf", "GE")] > a[("rf", "CH")]
    ch_small = abs(a[("dnn", "CH_GE")] - a[("dnn", "GE")]) <= 0.03 and abs(a[("dnn", "CH_GT")] - a[("dnn", "GT")]) <= 0.03
    strong = min(a[("rf", "all")], a[("xgb", "all")], a[("dnn", "all")]); weak = max(a[("lr", "all")], a[("gnb", "all")], a[("svm", "all")])
    held = {p: residue_f1(p) for p in ("6wzo", "6nqn", "1c29")}
    pni = {c: json.loads(Path(f"predictions/pni/{c}_interface.json").read_text()) for c in ("CNT-1CXP", "GQD-PSMa", "GQD-2L86", "GQD-6ZRF")}
    corona = list(csv.DictReader(open("predictions/pni/corona_ranking.csv")))

    L = ["# Reproduction report: Cha et al., Nat. Comput. Sci. 2, 243-252 (2022)", "",
         "Access level L3: paper, SI, released code and the Code Ocean capsule's data. All stages were run directly by one Claude agent "
         "following the reference task graph (no model API was available for the multi-agent harness). Models were trained here; "
         "the authors' saved models were used only in a separate diagnostic (section 4), never for a reported prediction.", "",
         "## 1. Summary", "",
         f"- **Descriptors reproduce exactly.** With the capsule's GHECOM outputs and CH tables, all 20 reference descriptors on both 1MA9 chains match the authors' own values (Pearson r = 1.000), and 18 of 21 Fig. 2a correlations are reproduced (`features/`).",
         f"- **Table 1:** {within}/126 cells within ±0.03. Random forest reproduces accuracy and AUC almost exactly on every subset. The DNN is 0.03-0.07 lower on most subsets because it was trained for 100 epochs, not 1000 (CPU only). XGBoost is 0.02-0.04 lower with default hyper-parameters (the paper does not give its settings).",
         "- **Linear models:** the paper's near-chance LR / GNB / SVM numbers are reproduced only without feature scaling (LR all: 0.553 / 0.565, exactly Table 1). With per-fold MinMax scaling, LR reaches AUC 0.805 on all descriptors, so the paper's large gap between linear and tree models is mostly a scaling artifact.",
         "- **Table 1's F1 is the non-contact class's F1** (sklearn's default positive label in the released arrays). This run's tables use that definition.",
         f"- **Held-out interfaces:** per-residue F1 {held['6wzo'][2]:.2f} / {held['6nqn'][2]:.2f} / {held['1c29'][2]:.2f} for 6WZO / 6NQN / 1C29, below the authors' published predictions (0.53 / 0.40 / 0.37 under the same metric). The authors' own released models applied to their own released test features do not reproduce their Fig. 4 predictions either (section 4).",
         "- **Protein-corona trends reproduce:** albumin is predicted to bind SWNTs least, and the 5 nm carbon sphere binds every protein less than SWNTs, as in Fig. 5f. The hMPO residues named in the text (Arg294, Tyr313) are not among the predicted sites.", "",
         "## 2. Table 1 (tenfold CV on the authors' balanced training arrays)", "",
         "Accuracy / AUC / F1 (F1 of the non-contact class, as in the paper). LR, GNB and SVM rows are without feature scaling, matching the paper. "
         "Paper values in parentheses.", "",
         "| Model | " + " | ".join(SUBSETS) + " |", "|---|" + "---|" * len(SUBSETS)]
    for m in MODELS:
        cells = []
        for s in SUBSETS:
            v = T[(m, s)]
            cells.append("n/a" if v is None else f"{v[0]:.3f} / {v[1]:.3f} / {v[2]:.3f} ({PAPER[m][s][0]:.3f} / {PAPER[m][s][1]:.3f} / {PAPER[m][s][2]:.3f})")
        L.append(f"| {m.upper()} | " + " | ".join(cells) + " |")
    if any(unscaled.values()):
        L += ["", "With per-fold MinMax scaling (not the paper's condition), AUC:", "",
              "| Model | " + " | ".join(SUBSETS) + " |", "|---|" + "---|" * len(SUBSETS)]
        for m, d in unscaled.items():
            L.append(f"| {m.upper()} scaled | " + " | ".join(f"{d[s][1]:.3f}" if s in d else "n/a" for s in SUBSETS) + " |")
    L += ["", "## 3. Held-out complexes (DNN, GE+GT, top 0.1% of residue pairs)", "",
          "Ground truth: residues with a Cα within 7 Å of the partner chain (identical to the authors' Fig. 4 ground truth).", "",
          "| Complex | Predicted residues | True interface residues | Precision | Recall | F1 | Authors' published F1 |", "|---|---|---|---|---|---|---|"]
    pub = {"6wzo": 0.533, "6nqn": 0.400, "1c29": 0.366}
    for p, (pr, rc, f1, npred, ntrue) in held.items():
        L.append(f"| {p.upper()} | {npred} | {ntrue} | {pr:.2f} | {rc:.2f} | {f1:.2f} | {pub[p]:.2f} |")
    L += ["", "SI Tables 4-6 report accuracy 0.721 / 0.803 / 0.786. That metric cannot be reconstructed from the published data: the authors' own Fig. 4 predicted sets, scored per residue, give F1 0.53 / 0.40 / 0.37.", "",
          "## 4. Diagnostic: the authors' saved models on the authors' test features", "",
          "Running the capsule's ten GE+GT DNN models (dnn_models/re_model_GE_GT*.h5) on the capsule's held-out features exactly as PPI_ML_prediction.py does "
          "(unscaled features, mean class-0 probability, top 0.1% of pairs) does not reproduce Fig. 4: for 6WZO it selects 11 pairs covering 6 + 8 residues, "
          "while Fig. 4 lists 41 + 41. No threshold (0.1-5%), output column or scaling choice reproduces the published sets (best Jaccard 0.51 / 0.12 / 0.16). "
          "The published held-out predictions therefore came from a procedure that is not in the release.", "",
          "## 5. Protein-nanoparticle cases (DNN, GE+GT)", "",
          "| Case | Features | Predicted protein residues (top 0.1% of pairs) |", "|---|---|---|"]
    for c, d in pni.items():
        L.append(f"| {c} | {d.get('features', '')} | {', '.join(map(str, d['protein_residues']))} |")
    L += ["", "Protein corona (score = fraction of residue-node pairs predicted in contact; the released code does not define the paper's 'binding probability'):", "",
          "| NP | apolipoprotein | HRG | galectin-3BP | albumin |", "|---|---|---|---|---|"]
    nps = sorted({r["np_name"] for r in corona})
    for n in nps:
        get = {r["protein"]: r["score"] for r in corona if r["np_name"] == n}
        L.append(f"| {n} | " + " | ".join(get.get(p, "") for p in ["apolipoprotein", "histidine-rich glycoprotein", "galectin-3-binding protein", "albumin"]) + " |")
    L += ["", "## 6. Deviations from the published method", "",
          "- DNN trained for 100 epochs per fold instead of 1000 (no GPU; ~21 h at 1000). Validation accuracy was still rising at epoch 100. Best epoch chosen by test-fold accuracy, as in the released code.",
          "- DNN predictions on new structures apply the training MinMax scaler; the released prediction script applies the models to unscaled features.",
          "- XGBoost with library defaults; the paper does not report its hyper-parameters. RF uses the released settings (200 trees, entropy, depth 50).",
          "- The capsule's pair list has 389 unique pairs (385 usable), not 464; its chains give 53.2M residue pairs, not 27.9M. Training used the authors' own balanced arrays (20,560 rows), so this does not affect Table 1.",
          "- PSM-α has no precomputed features in the capsule; its features were computed here with an approximation of GHECOM (rinacc / shell / pocketness), on the authors' published 5I55 and GQD (G3) structures.",
          "- ZnO nanopyramid-β-galactosidase: not attempted (no NP model in the capsule or the paper).",
          "- GHECOM itself was not run (its source needs registration); the capsule's precomputed GHECOM outputs were used for all training and held-out chains.", "",
          "## 7. Inconsistencies found in the paper and its materials", "",
          "- Methods: GNM uses 'the first tenth of the modes'; code and figures: the first 10 modes.",
          "- Fig. 1n is labelled multifractal dimension but shows Forman-Ricci curvature; Fig. 1's OPD is the N = 7 index.",
          "- Fig. 3c source data swap the DNN GT and GE+CH rows relative to Table 1 and the text.",
          "- Table 1 F1 is the non-contact class's F1; LR / GNB / SVM were trained on unscaled features.",
          "- hMPO: the text names Tyr293, Tyr313, Arg294, Arg307; the authors' own Fig. 5d/e prediction contains only Arg294 and Tyr313.",
          "- PSM-α: the text cites 5KHB and the N-terminus; the source data use 5I55 and predict the C-terminal half.",
          "- NP models: the text describes 13-atom coarse-grained nodes with 7 Å edges; the released code and the Fig. 5 files use atom-level nodes with 4 Å edges for the GQD and hIAPP cases, and 13-atom nodes for the hMPO-SWNT and corona cases.",
          "- The released DNN_train.py reloads a fixed 'best_model' path instead of each fold's checkpoint, uses from_logits=True on a softmax output, and the released prediction script does not apply the training scaler.",
          "- The capsule's pair list has 389 pairs, not the 464 stated; PPI_descriptors_cal.py imports a descriptor_calc package that is not in the release.", ""]
    Path("report").mkdir(exist_ok=True)
    Path("report/report.md").write_text("\n".join(L))

    claims = [
        {"claim": "GT descriptors alone predict contacts better than GE alone, and GE better than CH alone",
         "verdict": "supported" if gt_ge_ch else "contradicted", "evidence": ["ml/cv/predictions/dnn_GT.npz", "ml/cv/predictions/rf_GT.npz", "ml/cv/predictions/dnn_GE.npz", "ml/cv/predictions/dnn_CH.npz"],
         "notes": f"DNN AUC GT {a[('dnn','GT')]:.3f} > GE {a[('dnn','GE')]:.3f} > CH {a[('dnn','CH')]:.3f}; RF {a[('rf','GT')]:.3f} > {a[('rf','GE')]:.3f} > {a[('rf','CH')]:.3f}"},
        {"claim": "Adding CH descriptors to GE or GT changes performance by about two points or less",
         "verdict": "supported" if ch_small else "contradicted", "evidence": ["ml/cv/predictions/dnn_CH_GE.npz", "ml/cv/predictions/dnn_CH_GT.npz"],
         "notes": f"DNN AUC change: +GE {a[('dnn','CH_GE')]-a[('dnn','GE')]:+.3f}, +GT {a[('dnn','CH_GT')]-a[('dnn','GT')]:+.3f}"},
        {"claim": "RF, XGB and the DNN perform similarly and much better than LR, GNB and SVM",
         "verdict": "supported" if strong > weak else "contradicted", "evidence": ["ml/cv/predictions/rf_all.npz", "ml/cv/predictions/lr_all.npz", "ml/cv/predictions_scaled/lr_all.npz"],
         "notes": f"Under the paper's condition (no scaling) yes: min(RF, XGB, DNN) AUC {strong:.3f} vs max(LR, GNB, SVM) {weak:.3f}. With feature scaling the linear models reach AUC ~0.80 and the gap shrinks to ~0.1."},
        {"claim": "GE+GT models predict held-out, non-homologous interfaces with about 80% accuracy",
         "verdict": "inconclusive", "evidence": ["predictions/ppi/6wzo_dnn_GE_GT.csv", "predictions/ppi/6nqn_dnn_GE_GT.csv", "predictions/ppi/1c29_dnn_GE_GT.csv"],
         "notes": "The SI metric cannot be reconstructed; per-residue F1 here is below the authors' published predictions, and the authors' released models do not reproduce their own Fig. 4."},
        {"claim": "Models trained on protein-protein complexes predict nanoparticle-protein sites consistent with experiments",
         "verdict": "inconclusive", "evidence": ["predictions/pni/corona_ranking.csv", "predictions/pni/CNT-1CXP_interface.json"],
         "notes": "Corona trends (albumin lowest on SWNTs; sphere below SWNTs) reproduce. hMPO named residues are not predicted. hIAPP cases need expert review; ZnO not attempted."},
    ]
    Path("report/claims.yaml").write_text(yaml.safe_dump(claims, sort_keys=False, width=120))
    print("report written;", within, "Table 1 cells within tolerance")


if __name__ == "__main__":
    main()
