# Reproduction report: Cha et al., Nat. Comput. Sci. 2, 243-252 (2022)

Access level L3: paper, SI, released code and the Code Ocean capsule's data. All stages were run directly by one Claude agent following the reference task graph (no model API was available for the multi-agent harness). Models were trained here; the authors' saved models were used only in a separate diagnostic (section 4), never for a reported prediction.

## 1. Summary

- **Descriptors reproduce exactly.** With the capsule's GHECOM outputs and CH tables, all 20 reference descriptors on both 1MA9 chains match the authors' own values (Pearson r = 1.000), and 18 of 21 Fig. 2a correlations are reproduced (`features/`).
- **Table 1:** 96/126 cells within ±0.03. Logistic regression, naive Bayes, SVM and random forest reproduce Table 1 to the third decimal on almost every cell, F1 included. The DNN is 0.03-0.07 lower on most subsets because it was trained for 100 epochs, not 1000 (CPU only); its CH-only row matches. XGBoost is 0.03-0.05 lower with library defaults (the paper does not give its hyper-parameters).
- **Linear models:** the paper's near-chance LR / GNB / SVM numbers are reproduced only without feature scaling (LR all: 0.553 / 0.565, exactly Table 1). With per-fold MinMax scaling, LR reaches AUC 0.805 on all descriptors, so the paper's large gap between linear and tree models is mostly a scaling artifact.
- **Table 1's F1 is the non-contact class's F1** (sklearn's default positive label in the released arrays). This run's tables use that definition.
- **Held-out interfaces:** per-residue F1 0.21 / 0.16 / 0.18 for 6WZO / 6NQN / 1C29, below the authors' published predictions (0.53 / 0.40 / 0.37 under the same metric). The authors' own released models applied to their own released test features do not reproduce their Fig. 4 predictions either (section 4).
- **Protein-corona trends reproduce:** albumin is predicted to bind SWNTs least, and the 5 nm carbon sphere binds every protein less than SWNTs, as in Fig. 5f. The hMPO residues named in the text (Arg294, Tyr313) are not among the predicted sites.

## 2. Table 1 (tenfold CV on the authors' balanced training arrays)

Accuracy / AUC / F1 (F1 of the non-contact class, as in the paper). LR, GNB and SVM rows are without feature scaling, matching the paper. Paper values in parentheses.

| Model | all | CH | GE | GT | GE_GT | CH_GE | CH_GT |
|---|---|---|---|---|---|---|---|
| LR | 0.553 / 0.565 / 0.592 (0.553 / 0.565 / 0.592) | 0.551 / 0.574 / 0.551 (0.551 / 0.573 / 0.553) | 0.553 / 0.563 / 0.592 (0.553 / 0.563 / 0.592) | 0.674 / 0.733 / 0.650 (0.671 / 0.730 / 0.648) | 0.551 / 0.562 / 0.587 (0.551 / 0.562 / 0.587) | 0.551 / 0.562 / 0.589 (0.551 / 0.562 / 0.589) | 0.691 / 0.753 / 0.675 (0.684 / 0.743 / 0.667) |
| GNB | 0.578 / 0.633 / 0.668 (0.578 / 0.633 / 0.668) | 0.537 / 0.559 / 0.480 (0.537 / 0.559 / 0.480) | 0.574 / 0.628 / 0.665 (0.574 / 0.628 / 0.665) | 0.615 / 0.711 / 0.670 (0.615 / 0.711 / 0.670) | 0.576 / 0.630 / 0.666 (0.576 / 0.630 / 0.666) | 0.575 / 0.627 / 0.666 (0.575 / 0.627 / 0.666) | 0.652 / 0.709 / 0.668 (0.652 / 0.709 / 0.668) |
| SVM | 0.607 / 0.656 / 0.655 (0.607 / 0.656 / 0.655) | 0.552 / 0.576 / 0.523 (0.552 / 0.576 / 0.523) | 0.606 / 0.652 / 0.655 (0.606 / 0.652 / 0.655) | 0.663 / 0.719 / 0.589 (0.663 / 0.719 / 0.589) | 0.606 / 0.652 / 0.657 (0.606 / 0.652 / 0.657) | 0.605 / 0.650 / 0.656 (0.605 / 0.650 / 0.656) | 0.674 / 0.741 / 0.607 (0.674 / 0.741 / 0.607) |
| RF | 0.822 / 0.920 / 0.803 (0.822 / 0.920 / 0.803) | 0.561 / 0.583 / 0.545 (0.561 / 0.583 / 0.545) | 0.769 / 0.869 / 0.737 (0.769 / 0.869 / 0.737) | 0.804 / 0.889 / 0.789 (0.804 / 0.889 / 0.789) | 0.810 / 0.907 / 0.789 (0.810 / 0.907 / 0.789) | 0.802 / 0.899 / 0.778 (0.802 / 0.899 / 0.778) | 0.812 / 0.905 / 0.797 (0.812 / 0.905 / 0.797) |
| XGB | 0.804 / 0.883 / 0.790 (0.829 / 0.913 / 0.817) | 0.560 / 0.583 / 0.544 (0.559 / 0.583 / 0.542) | 0.739 / 0.820 / 0.713 (0.780 / 0.862 / 0.763) | 0.768 / 0.844 / 0.751 (0.804 / 0.879 / 0.794) | 0.786 / 0.865 / 0.768 (0.815 / 0.899 / 0.803) | 0.770 / 0.848 / 0.752 (0.798 / 0.885 / 0.784) | 0.779 / 0.857 / 0.765 (0.815 / 0.895 / 0.804) |
| DNN | 0.795 / 0.852 / 0.782 (0.840 / 0.908 / 0.830) | 0.572 / 0.587 / 0.536 (0.574 / 0.590 / 0.548) | 0.736 / 0.798 / 0.719 (0.770 / 0.838 / 0.756) | 0.750 / 0.810 / 0.735 (0.808 / 0.877 / 0.800) | 0.784 / 0.847 / 0.767 (0.823 / 0.891 / 0.811) | 0.766 / 0.823 / 0.747 (0.791 / 0.857 / 0.768) | 0.758 / 0.812 / 0.744 (0.806 / 0.876 / 0.794) |

With per-fold MinMax scaling (not the paper's condition), AUC:

| Model | all | CH | GE | GT | GE_GT | CH_GE | CH_GT |
|---|---|---|---|---|---|---|---|
| LR scaled | 0.805 | 0.574 | 0.764 | 0.735 | 0.782 | 0.791 | 0.755 |
| GNB scaled | 0.775 | 0.559 | 0.753 | 0.711 | 0.759 | 0.760 | 0.709 |
| SVM scaled | 0.819 | 0.576 | 0.776 | 0.764 | 0.807 | 0.801 | 0.766 |

## 3. Held-out complexes (DNN, GE+GT, top 0.1% of residue pairs)

Ground truth: residues with a Cα within 7 Å of the partner chain (identical to the authors' Fig. 4 ground truth).

| Complex | Predicted residues | True interface residues | Precision | Recall | F1 | Authors' published F1 |
|---|---|---|---|---|---|---|
| 6WZO | 12 | 83 | 0.83 | 0.12 | 0.21 | 0.53 |
| 6NQN | 35 | 26 | 0.14 | 0.19 | 0.16 | 0.40 |
| 1C29 | 62 | 39 | 0.15 | 0.23 | 0.18 | 0.37 |

SI Tables 4-6 report accuracy 0.721 / 0.803 / 0.786. That metric cannot be reconstructed from the published data: the authors' own Fig. 4 predicted sets, scored per residue, give F1 0.53 / 0.40 / 0.37.

## 4. Diagnostic: the authors' saved models on the authors' test features

Running the capsule's ten GE+GT DNN models (dnn_models/re_model_GE_GT*.h5) on the capsule's held-out features exactly as PPI_ML_prediction.py does (unscaled features, mean class-0 probability, top 0.1% of pairs) does not reproduce Fig. 4: for 6WZO it selects 11 pairs covering 6 + 8 residues, while Fig. 4 lists 41 + 41. No threshold (0.1-5%), output column or scaling choice reproduces the published sets (best Jaccard 0.51 / 0.12 / 0.16). The published held-out predictions therefore came from a procedure that is not in the release.

## 5. Protein-nanoparticle cases (DNN, GE+GT)

| Case | Features | Predicted protein residues (top 0.1% of pairs) |
|---|---|---|
| CNT-1CXP | capsule data_feature_1cxp_AC_ASWNT_14_0_500A_13_reduced.npy | 2, 3, 4, 5, 6, 33, 52, 103, 104, 155, 216, 487 |
| GQD-PSMa | hybrid: N4 pipeline with GHECOM approximation on 5I55 + G3 | 20 |
| GQD-2L86 | capsule data_feature_2l86_Aiapp_gqd_atomic.npy | 1, 4, 6, 37 |
| GQD-6ZRF | capsule data_feature_6zrf_combined_Aiapp_gqd_atomic.npy | 13, 14, 15, 16, 17, 20, 27, 28, 30, 109, 133, 135, 137, 139, 140, 141, 142, 148, 150, 157, 158, 159, 160, 161, 164, 165, 170, 172, 174, 181, 279, 292 |

Protein corona (score = fraction of residue-node pairs predicted in contact; the released code does not define the paper's 'binding probability'):

| NP | apolipoprotein | HRG | galectin-3BP | albumin |
|---|---|---|---|---|
| SWNT_62_50nm_5A | 0.4902 | 0.4154 | 0.4677 | 0.2597 |
| SWNT_66_50nm_8A | 0.9603 | 0.5621 | 0.4876 | 0.0158 |
| SWNT_78_50nm_1nm | 0.9834 | 0.6481 | 0.5908 | 0.011 |
| SWNT_87_50nm_1nm | 0.9837 | 0.6528 | 0.5974 | 0.0136 |
| spherical_C_5nm | 0.0097 | 0.0112 | 0.0092 | 0.0061 |

## 6. Deviations from the published method

- DNN trained for 100 epochs per fold instead of 1000 (no GPU; ~21 h at 1000). Validation accuracy was still rising at epoch 100. Best epoch chosen by test-fold accuracy, as in the released code.
- DNN predictions on new structures apply the training MinMax scaler; the released prediction script applies the models to unscaled features.
- XGBoost with library defaults; the paper does not report its hyper-parameters. RF uses the released settings (200 trees, entropy, depth 50).
- The capsule's pair list has 389 unique pairs (385 usable), not 464; its chains give 53.2M residue pairs, not 27.9M. Training used the authors' own balanced arrays (20,560 rows), so this does not affect Table 1.
- PSM-α has no precomputed features in the capsule; its features were computed here with an approximation of GHECOM (rinacc / shell / pocketness), on the authors' published 5I55 and GQD (G3) structures.
- ZnO nanopyramid-β-galactosidase: not attempted (no NP model in the capsule or the paper).
- GHECOM itself was not run (its source needs registration); the capsule's precomputed GHECOM outputs were used for all training and held-out chains.

## 7. Inconsistencies found in the paper and its materials

- Methods: GNM uses 'the first tenth of the modes'; code and figures: the first 10 modes.
- Fig. 1n is labelled multifractal dimension but shows Forman-Ricci curvature; Fig. 1's OPD is the N = 7 index.
- Fig. 3c source data swap the DNN GT and GE+CH rows relative to Table 1 and the text.
- Table 1 F1 is the non-contact class's F1; LR / GNB / SVM were trained on unscaled features.
- hMPO: the text names Tyr293, Tyr313, Arg294, Arg307; the authors' own Fig. 5d/e prediction contains only Arg294 and Tyr313.
- PSM-α: the text cites 5KHB and the N-terminus; the source data use 5I55 and predict the C-terminal half.
- NP models: the text describes 13-atom coarse-grained nodes with 7 Å edges; the released code and the Fig. 5 files use atom-level nodes with 4 Å edges for the GQD and hIAPP cases, and 13-atom nodes for the hMPO-SWNT and corona cases.
- The released DNN_train.py reloads a fixed 'best_model' path instead of each fold's checkpoint, uses from_logits=True on a softmax output, and the released prediction script does not apply the training scaler.
- The capsule's pair list has 389 pairs, not the 464 stated; PPI_descriptors_cal.py imports a descriptor_calc package that is not in the release.
