# Evaluation: opus-direct-L2

Access level **L2**, plan mode **direct**, roles **single-agent (Claude in the coding session, acting as every worker role)**.

**Reproduction score: 55.3 / 100**

| Checkpoint | Status | Score | Weight | Notes |
|---|---|---|---|---|
| C0 Environment and tools run | fail | 0.62 | 5 | 5/8 smoke tests passed |
| C1 Method spec extracted | pass | 1.00 | 10 | 22/22 checklist items correct (pass at 85%) |
| C2 Training data rebuilt | fail | 0.33 | 10 | training pairs: 200 (paper: 464) |
| C3 Descriptors correct | fail | 0.71 | 20 | Fig. 2a correlations (|r| >= 0.3 in the paper): 16/21 reproduced within ±0.25 with the same sign, over 88642 residues |
| C4 Table 1 reproduced | fail | 0.78 | 25 | 97/126 Table 1 cells within ±0.03 of the paper |
| C5 Held-out PPI interfaces | fail | 0.33 | 15 | 6WZO: per-residue F1 0.151 (precision 0.70, recall 0.08; authors 0.533, pass >= 0.433; recomputed from pair probabilities); overlap with the authors' predicted residues (Jaccard) 0.07 |
| C6 Protein-NP predictions | fail | 0.00 | 15 | 0/6 findings matched, 2 pending review (pass at 5) |
| C7 Report and claim verdicts | needs_review | 0.00 | 0 | Score the report 1-5 on fidelity, execution, accuracy, completeness, rigor in eval/c7_scores.yaml (by a reviewer or `mathbac judge`). |

## Details

### C0 Environment and tools run
- 5/8 smoke tests passed
- ghecom: missing
- unified_ppi_descriptors: failed
- unified_pni_descriptors: failed
- Self-reported by the agent; rerun the card smoke tests in the image to confirm.

### C1 Method spec extracted
- 22/22 checklist items correct (pass at 85%)

### C2 Training data rebuilt
- training pairs: 200 (paper: 464)
- residue pairs recomputed from distance matrices: 10,208,383 (paper: 27,859,297; off by 63.4%)
- 1MA9 distance matrix max deviation from RCSB coordinates: 0.0000 Å

### C3 Descriptors correct
- Fig. 2a correlations (|r| >= 0.3 in the paper): 16/21 reproduced within ±0.25 with the same sign, over 88642 residues
- 14/20 reference descriptors match (Pearson r >= 0.99, per-column overrides none)
- r by descriptor: c_charge 0.994, hydrophobicity 1.000, orc 1.000, frc 1.000, gnm 1.000, opd_7 1.000, mfd 1.000, rinacc 0.944, shell 0.887, pocketness 0.321, c_charge 0.995, hydrophobicity 1.000, orc 1.000, frc 1.000, gnm 1.000, opd_7 1.000, mfd 1.000, rinacc 0.955, shell 0.902, pocketness 0.220

### C4 Table 1 reproduced
- 97/126 Table 1 cells within ±0.03 of the paper
- headline cells (DNN GT, GE_GT, all, CH; RF/XGB all) not all within tolerance
- orderings: gt_gt_ge=yes, ge_gt_ch=yes, tree_vs_linear=yes, ch_adds_little_ge=no, ch_adds_little_gt=yes

### C5 Held-out PPI interfaces
- 6WZO: per-residue F1 0.151 (precision 0.70, recall 0.08; authors 0.533, pass >= 0.433; recomputed from pair probabilities); overlap with the authors' predicted residues (Jaccard) 0.07
- 6NQN: per-residue F1 0.127 (precision 0.11, recall 0.15; authors 0.400, pass >= 0.300; recomputed from pair probabilities); overlap with the authors' predicted residues (Jaccard) 0.13
- 1C29: per-residue F1 0.268 (precision 0.22, recall 0.33; authors 0.366, pass >= 0.266; recomputed from pair probabilities); overlap with the authors' predicted residues (Jaccard) 0.11
- SI Tables 4-6 report accuracy 0.721 / 0.803 / 0.786 under a definition the published data do not allow us to reconstruct.

### C6 Protein-NP predictions
- 0/6 findings matched, 2 pending review (pass at 5)
- cnt_hmpo: required ARG294, TYR313: 0 predicted; residues named in the text that were predicted: none
- gqd_psma: 0% of predicted residues in 11-22 (pass at 50%)
- corona: no - On SWNTs, albumin ranks below apolipoprotein, HRG and galectin-3BP (Fig. 5f: 0.78-0.80 vs 0.93-0.99)
- spherical_lower: no - The 5 nm spherical carbon NP scores below SWNTs for every corona protein (Fig. 5f: 0.64-0.77 vs 0.78-0.99)
- gqd_iapp: needs review - Hydroxylated GQD-hIAPP monomer (2L86) sites match LC-MS/MS (Faridi et al.) and fluorescence quenching (Wang et al.)
- zno_bgal: needs review - ZnO nanopyramid binds β-galactosidase at the apex and edges of the pyramid

### C7 Report and claim verdicts
- Score the report 1-5 on fidelity, execution, accuracy, completeness, rigor in eval/c7_scores.yaml (by a reviewer or `mathbac judge`).

## Communication

- Fact coverage (facts passed to at least one node): 0%; fact accuracy: n/a
- Facts never passed: carbon_atoms_per_node, contact_cutoff_A, cv_folds, gnm_modes_fraction, graph_edge_cutoff_A, interface_threshold_top_fraction, node_curvature_aggregation, opd_neighbourhood_sizes, residue_representation, training_set_n_pairs, zno_atoms_per_node
- Grounding ratio: n/a
- Contradiction rate: n/a
- Tool error rate: n/a
- Verifier rejections: none
- Undetected faults (accepted in the loop, failed in evaluation): none
- Cost: $0, 0 model turns, 0.0 min wall clock
