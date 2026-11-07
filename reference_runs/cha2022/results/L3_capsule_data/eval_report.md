# Evaluation: opus-direct-L3

Access level **L3**, plan mode **direct**, roles **single-agent (Claude in the coding session, acting as every worker role)**.

**Reproduction score: 65.4 / 100**

| Checkpoint | Status | Score | Weight | Notes |
|---|---|---|---|---|
| C0 Environment and tools run | fail | 0.62 | 5 | 5/8 smoke tests passed |
| C1 Method spec extracted | pass | 1.00 | 10 | 22/22 checklist items correct (pass at 85%) |
| C2 Training data rebuilt | fail | 0.33 | 10 | training pairs: 385 (paper: 464) |
| C3 Descriptors correct | pass | 0.97 | 20 | Fig. 2a correlations (|r| >= 0.3 in the paper): 18/21 reproduced within ±0.25 with the same sign, over 163812 residues |
| C4 Table 1 reproduced | fail | 0.88 | 25 | 96/126 Table 1 cells within ±0.03 of the paper |
| C5 Held-out PPI interfaces | fail | 0.00 | 15 | 6WZO: per-residue F1 0.227 (precision 0.79, recall 0.13; authors 0.533, pass >= 0.433; recomputed from pair probabilities); overlap with the authors' predicted residues (Jaccard) 0.09 |
| C6 Protein-NP predictions | needs_review | 0.50 | 15 | 3/6 findings matched, 2 pending review (pass at 5) |
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
- training pairs: 385 (paper: 464)
- residue pairs recomputed from distance matrices: 53,157,357 (paper: 27,859,297; off by 90.8%)
- 1MA9 distance matrix max deviation from RCSB coordinates: 0.0000 Å

### C3 Descriptors correct
- Fig. 2a correlations (|r| >= 0.3 in the paper): 18/21 reproduced within ±0.25 with the same sign, over 163812 residues
- 20/20 reference descriptors match (Pearson r >= 0.99, per-column overrides none)
- r by descriptor: c_charge 1.000, hydrophobicity 1.000, orc 1.000, frc 1.000, gnm 1.000, opd_7 1.000, mfd 1.000, rinacc 1.000, shell 1.000, pocketness 1.000, c_charge 1.000, hydrophobicity 1.000, orc 1.000, frc 1.000, gnm 1.000, opd_7 1.000, mfd 1.000, rinacc 1.000, shell 1.000, pocketness 1.000

### C4 Table 1 reproduced
- 96/126 Table 1 cells within ±0.03 of the paper
- headline cells (DNN GT, GE_GT, all, CH; RF/XGB all) not all within tolerance
- orderings: gt_gt_ge=yes, ge_gt_ch=yes, tree_vs_linear=yes, ch_adds_little_ge=yes, ch_adds_little_gt=yes

### C5 Held-out PPI interfaces
- 6WZO: per-residue F1 0.227 (precision 0.79, recall 0.13; authors 0.533, pass >= 0.433; recomputed from pair probabilities); overlap with the authors' predicted residues (Jaccard) 0.09
- 6NQN: per-residue F1 0.138 (precision 0.12, recall 0.15; authors 0.400, pass >= 0.300; recomputed from pair probabilities); overlap with the authors' predicted residues (Jaccard) 0.17
- 1C29: per-residue F1 0.176 (precision 0.14, recall 0.23; authors 0.366, pass >= 0.266; recomputed from pair probabilities); overlap with the authors' predicted residues (Jaccard) 0.08
- SI Tables 4-6 report accuracy 0.721 / 0.803 / 0.786 under a definition the published data do not allow us to reconstruct.

### C6 Protein-NP predictions
- 3/6 findings matched, 2 pending review (pass at 5)
- cnt_hmpo: required ARG294, TYR313: 0 predicted; residues named in the text that were predicted: none
- gqd_psma: 100% of predicted residues in 11-22 (pass at 50%)
- corona: yes - On SWNTs, albumin ranks below apolipoprotein, HRG and galectin-3BP (Fig. 5f: 0.78-0.80 vs 0.93-0.99)
- spherical_lower: yes - The 5 nm spherical carbon NP scores below SWNTs for every corona protein (Fig. 5f: 0.64-0.77 vs 0.78-0.99)
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
