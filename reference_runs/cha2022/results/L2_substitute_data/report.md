# Reproduction report: Cha et al., Nat. Comput. Sci. 2, 243-252 (2022)

Access level L2: paper, SI and released code, without the Code Ocean capsule's data. All stages were run directly by one Claude agent following the reference task graph (no model API was available for the multi-agent harness).

Because the paper's 464 training pairs are listed only in the capsule, the training set here is a documented substitute: 200 X-ray heterodimers curated from RCSB (resolution <= 2.5 Å, one entry per UniProt pair, < 30% identity to the held-out proteins; `scripts/n3_curate.py`). CH charges were recomputed with PDB2PQR (CHARMM); GHECOM needs registration, so its three pocket descriptors are approximated by multiscale grid morphology (`scripts/n4_descriptors.py`). The other descriptors (OPD, GNM, ORC, FRC, MFD) use the released code. Numbers are therefore expected to differ from Table 1; the question is whether the paper's qualitative findings hold on independent data.

## 1. Summary

- **Table 1:** 97/126 computed cells within ±0.03 of the published values, on a different training set.
- **Descriptor ranking:** DNN AUC GT 0.836, GE 0.785, CH 0.612; RF GT 0.903, GE 0.845, CH 0.608. GT > GE > CH holds as in the paper.
- **Adding CH:** DNN AUC change +0.049 (to GE) and +0.002 (to GT). to GE: outside the paper's ±0.03; to GT: within the paper's ±0.03.
- **Tree / DNN vs linear models (unscaled, the paper's condition):** min(RF, XGB, DNN) AUC 0.874 vs max(LR, GNB, SVM) 0.677; with per-fold MinMax scaling LR reaches 0.813.
- **Held-out interfaces:** per-residue F1 0.15 / 0.13 / 0.27 for 6WZO / 6NQN / 1C29 (authors' published predictions: 0.53 / 0.40 / 0.37 under the same metric).
- **Protein-nanoparticle cases:** hMPO-SWNT and the hIAPP / PSM-α GQD cases were run from the Fig. 5 source-data structures; the protein-corona ranking and ZnO-β-galactosidase need NP models that are only in the capsule or not released.

## 2. Table 1 (tenfold CV on the substitute training set)

Accuracy / AUC / F1 (F1 of the non-contact class, as in the paper). LR, GNB and SVM rows are without feature scaling, matching the paper. Paper values in parentheses.

| Model | all | CH | GE | GT | GE_GT | CH_GE | CH_GT |
|---|---|---|---|---|---|---|---|
| LR | 0.531 / 0.565 / 0.476 (0.553 / 0.565 / 0.592) | 0.569 / 0.596 / 0.566 (0.551 / 0.573 / 0.553) | 0.531 / 0.565 / 0.476 (0.553 / 0.563 / 0.592) | 0.650 / 0.713 / 0.634 (0.671 / 0.730 / 0.648) | 0.531 / 0.565 / 0.476 (0.551 / 0.562 / 0.587) | 0.531 / 0.565 / 0.476 (0.551 / 0.562 / 0.589) | 0.686 / 0.750 / 0.676 (0.684 / 0.743 / 0.667) |
| GNB | 0.597 / 0.677 / 0.664 (0.578 / 0.633 / 0.668) | 0.546 / 0.575 / 0.467 (0.537 / 0.559 / 0.480) | 0.596 / 0.675 / 0.664 (0.574 / 0.628 / 0.665) | 0.649 / 0.704 / 0.642 (0.615 / 0.711 / 0.670) | 0.598 / 0.677 / 0.664 (0.576 / 0.630 / 0.666) | 0.596 / 0.676 / 0.664 (0.575 / 0.627 / 0.666) | 0.665 / 0.715 / 0.650 (0.652 / 0.709 / 0.668) |
| SVM | 0.613 / 0.656 / 0.654 (0.607 / 0.656 / 0.655) | 0.541 / 0.576 / 0.615 (0.552 / 0.576 / 0.523) | 0.614 / 0.656 / 0.655 (0.606 / 0.652 / 0.655) | 0.659 / 0.710 / 0.603 (0.663 / 0.719 / 0.589) | 0.613 / 0.656 / 0.653 (0.606 / 0.652 / 0.657) | 0.613 / 0.656 / 0.653 (0.605 / 0.650 / 0.656) | 0.666 / 0.722 / 0.610 (0.674 / 0.741 / 0.607) |
| RF | 0.822 / 0.913 / 0.807 (0.822 / 0.920 / 0.803) | 0.576 / 0.608 / 0.570 (0.561 / 0.583 / 0.545) | 0.747 / 0.845 / 0.706 (0.769 / 0.869 / 0.737) | 0.813 / 0.903 / 0.802 (0.804 / 0.889 / 0.789) | 0.810 / 0.910 / 0.789 (0.810 / 0.907 / 0.789) | 0.783 / 0.870 / 0.762 (0.802 / 0.899 / 0.778) | 0.810 / 0.897 / 0.800 (0.812 / 0.905 / 0.797) |
| XGB | 0.829 / 0.906 / 0.820 (0.829 / 0.913 / 0.817) | 0.578 / 0.613 / 0.578 (0.559 / 0.583 / 0.542) | 0.748 / 0.829 / 0.725 (0.780 / 0.862 / 0.763) | 0.787 / 0.870 / 0.777 (0.804 / 0.879 / 0.794) | 0.809 / 0.891 / 0.798 (0.815 / 0.899 / 0.803) | 0.789 / 0.871 / 0.777 (0.798 / 0.885 / 0.784) | 0.798 / 0.878 / 0.791 (0.815 / 0.895 / 0.804) |
| DNN | 0.809 / 0.874 / 0.803 (0.840 / 0.908 / 0.830) | 0.586 / 0.612 / 0.597 (0.574 / 0.590 / 0.548) | 0.722 / 0.785 / 0.698 (0.770 / 0.838 / 0.756) | 0.769 / 0.836 / 0.759 (0.808 / 0.877 / 0.800) | 0.792 / 0.863 / 0.784 (0.823 / 0.891 / 0.811) | 0.766 / 0.834 / 0.756 (0.791 / 0.857 / 0.768) | 0.769 / 0.838 / 0.764 (0.806 / 0.876 / 0.794) |

With per-fold MinMax scaling (not the paper's condition), AUC:

| Model | all | CH | GE | GT | GE_GT | CH_GE | CH_GT |
|---|---|---|---|---|---|---|---|
| LR scaled | 0.813 | 0.595 | 0.743 | 0.714 | 0.772 | 0.789 | 0.751 |
| GNB scaled | 0.783 | 0.575 | 0.755 | 0.704 | 0.763 | 0.775 | 0.715 |
| SVM scaled | 0.842 | 0.599 | 0.762 | 0.736 | 0.804 | 0.818 | 0.769 |

## 3. Held-out complexes (DNN, GE+GT, top 0.1% of residue pairs)

Ground truth: residues with a Cα within 7 Å of the partner chain (identical to the authors' Fig. 4 ground truth).

| Complex | Predicted residues | True interface residues | Precision | Recall | F1 | Authors' published F1 |
|---|---|---|---|---|---|---|
| 6WZO | 10 | 83 | 0.70 | 0.08 | 0.15 | 0.53 |
| 6NQN | 37 | 26 | 0.11 | 0.15 | 0.13 | 0.40 |
| 1C29 | 58 | 39 | 0.22 | 0.33 | 0.27 | 0.37 |

SI Tables 4-6 report accuracy 0.721 / 0.803 / 0.786; that metric cannot be reconstructed from the published data.

## 4. Protein-nanoparticle cases (DNN, GE+GT)

| Case | Predicted protein residues (top 0.1% of pairs) |
|---|---|
| CNT-1CXP | 2, 4, 6, 33, 72, 113, 155 |
| GQD-PSMa | 2, 3 |
| GQD-2L86 | 3, 4, 36, 37 |
| GQD-6ZRF | 13, 16, 31, 32, 33, 35, 36, 37, 56, 60, 61, 84, 104, 108, 109, 128, 132, 133, 134, 136, 138, 143, 154, 155, 156, 157, 160, 175, 176, 177, 179, 180, 181, 204, 205, 228, 252, 253, 276, 277, 278, 279, 280, 281, 282, 294, 298, 299, 300 |

## 5. Deviations from the published method

- Training set: 200 substitute RCSB heterodimers instead of the paper's 464 IntAct/STRING pairs (not available without the capsule).
- GHECOM descriptors (rinacc, shell, pocketness) approximated by multiscale morphology; GHECOM itself needs registration.
- CH charges from PDB2PQR (CHARMM); residues PDB2PQR dropped (79 in 1EUV B) were filled with residue-type means.
- DNN trained for 100 epochs per fold instead of 1000 (CPU only). Best epoch chosen by test-fold accuracy, as in the released code.
- DNN predictions on new structures apply the training MinMax scaler; the released prediction script applies the models to unscaled features.
- XGBoost with library defaults; the paper does not report its hyper-parameters. RF uses the released settings (200 trees, entropy, depth 50).
- SVM without feature scaling uses the uncalibrated decision function through a logistic (same ranking, so the same AUC; Platt calibration on unscaled features was too slow). The scaled SVM uses Platt probabilities.
- NP graphs follow the released PNI_descriptors_cal.py (atom-level nodes, 4 Å edges), not the paper text (13-atom nodes, 7 Å).
- Protein corona and ZnO-β-galactosidase: not attempted (NP models not available at this access level).

## 6. Inconsistencies found in the paper and its materials

- Methods: GNM uses 'the first tenth of the modes'; code and figures: the first 10 modes.
- Fig. 1n is labelled multifractal dimension but shows Forman-Ricci curvature; Fig. 1's OPD is the N = 7 index.
- Fig. 3c source data swap the DNN GT and GE+CH rows relative to Table 1 and the text.
- hMPO: the text names Tyr293, Tyr313, Arg294, Arg307; the authors' own Fig. 5d/e prediction contains only Arg294 and Tyr313.
- PSM-α: the text cites 5KHB and the N-terminus; the source data use 5I55 and predict the C-terminal half.
- The released DNN_train.py reloads a fixed 'best_model' path instead of each fold's checkpoint, uses from_logits=True on a softmax output, and the released prediction script does not apply the training scaler.
- PPI_descriptors_cal.py imports a descriptor_calc package that is not in the release.
