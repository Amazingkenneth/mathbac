All paths are relative to the workspace (`/work`). CSV files have a header row. Residue numbers are the PDB author residue numbers (`resSeq`), with the insertion code appended when present (e.g. `52A`).

## spec/spec.yaml — method specification (node N1)

A YAML mapping. Every key below is required. Each value is a mapping with:

- `value`: the parameter value
- `source`: where it comes from, e.g. `paper p.245 "ML algorithms"`, `SI section 2.3.3`, `code: GNM.py:20`
- `status`: `stated` (paper or SI says it), `code_only` (only the code shows it), `unstated` (your proposed default), or `conflict` (paper and code disagree; give both in `value` as `{paper: ..., code: ...}`)

| Key | What it is |
|---|---|
| `residue_representation` | Atom that represents a residue, e.g. `CA` |
| `contact_cutoff_A` | Inter-chain distance below which a residue pair counts as in contact |
| `graph_edge_cutoff_A` | Intra-chain distance below which two residues share a graph edge |
| `descriptors` | Mapping `CH`, `GE`, `GT` to lists of descriptor names |
| `opd_neighbourhood_sizes` | Neighbourhood sizes used for the OPD chirality index |
| `gnm_modes_fraction` | Fraction of GNM modes kept in the square sum |
| `node_curvature_aggregation` | How edge curvatures become a node value |
| `mfd_features` | Which multifractal-dimension quantities are used per residue |
| `training_set` | Mapping with `n_pairs` and `n_residue_pairs` |
| `class_balancing` | How the contact / non-contact imbalance is handled |
| `descriptor_subsets` | The descriptor subsets models are trained on |
| `models` | The classifiers trained |
| `dnn` | Mapping with `layers`, `units`, `activation`, `dropout`, `output`, `loss` |
| `cv_folds` | Number of cross-validation folds |
| `metrics` | Metrics reported |
| `interface_threshold_top_fraction` | Fraction of residue pairs, by predicted probability, taken as the interface |
| `heldout_ppi` | PDB IDs of the held-out protein–protein complexes |
| `np_coarse_graining` | Mapping with `carbon_atoms_per_node`, `zno_atoms_per_node`, `np_graph_edge_cutoff_A` |
| `np_descriptor_subset` | Descriptor subset used for nanoparticle predictions |
| `extra` | Optional: any other parameters you found (free-form) |

## env/smoke_report.json (node N2)

```json
{"python": "3.8.x", "tools": [{"name": "ghecom", "status": "ok|failed|missing", "command": "...", "output": "..."}]}
```

One entry per installed tool card that this task uses. `status` is `ok` only if the card's smoke test passed.

## data/ — structures and labels (node N3)

- `data/pairs.csv` — columns `pdb_id,chain_a,chain_b,split` with `split` one of `train`, `heldout`. Training pairs from the paper's dataset; held-out rows for 6WZO, 6NQN and 1C29 (chains A and B).
- `data/substitutions.csv` — columns `requested_id,used_id,reason` (may be empty except for the header).
- `data/chains/<pdb>_<chain>.csv` — columns `resnum,resname,x,y,z` (Cα coordinates in Å), one row per residue in chain order. Every other per-chain file must use exactly these rows in this order.
- `data/distances/<pdb>_<chainA><chainB>.npz` — arrays `D` (shape `n_a × n_b`, Å), `res_a`, `res_b` (residue numbers as strings).
- `data/dataset_summary.json` — `{"n_pairs": int, "n_residue_pairs": int, "n_contacts": int}` over the `train` split.

## data/np/ — nanoparticles (node N3c)

- `data/np/<np_name>_cg.csv` — columns `node,x,y,z,n_atoms,element` for the coarse-grained nodes.
- `data/np/<np_name>_atoms.pdb` — the atomistic model the nodes came from.
- `data/np/cases.csv` — columns `case,np_name,pdb_id,chain,reference` listing each protein–nanoparticle case to predict.

## features/ — descriptors (nodes N4a–N4d)

- `features/CH/<pdb>_<chain>.csv`, `features/GE/<pdb>_<chain>.csv`, `features/GT/<pdb>_<chain>.csv`
- `features/np/<np_name>.csv` (GE and GT columns for nanoparticle nodes; first column `node`)

Protein feature files start with `resnum,resname` and have the same rows, in the same order, as `data/chains/<pdb>_<chain>.csv`. Required descriptor columns:

| Family | Required columns (extra columns are allowed) |
|---|---|
| CH | `charge`, `hydrophobicity`, `polarity`, `mw` |
| GE | `rinacc`, `shell`, `pocketness`, `opd_5`, `opd_7`, `opd_10`, `opd_15` |
| GT | `orc`, `frc`, `gnm`, `mfd`, `mfd_1`, `mfd_2`, `mfd_3`, `mfd_4` |

## ml/ — models (nodes N5, N6a, N6b)

- `ml/datasets/manifest.json` — per subset: `n_rows`, `n_features`, `feature_names`, `balancing`, `seed`.
- `ml/datasets/<subset>.npz` — arrays `X`, `y` (1 = contact) and `pair_index` (row → `pdb_chainAchainB:i:j`). Subsets are named `all`, `CH`, `GE`, `GT`, `GE_GT`, `CH_GE`, `CH_GT`.
- `ml/cv/predictions/<model>_<subset>.npz` — arrays `fold` (int), `y_true` (0/1), `y_score` (probability of contact), one entry per held-out row across all folds. Models are named `lr`, `gnb`, `svm`, `rf`, `xgb`, `dnn`.
- `ml/cv/metrics.csv` — columns `model,subset,fold,accuracy,auc,f1` (your own computation; the evaluator recomputes from the predictions).
- `ml/models/<model>_<subset>/` — saved final models.

## predictions/ (nodes N7, N8)

- `predictions/ppi/<pdb>_<model>_<subset>.csv` — columns `res_a,res_b,prob` for every residue pair of the held-out complex.
- `predictions/ppi/<pdb>_<model>_<subset>_interface.json` — `{"chain_a": [resnum, ...], "chain_b": [...], "top_fraction": 0.001}`.
- `predictions/pni/<case>.csv` — columns `np_node,res_protein,prob`.
- `predictions/pni/<case>_interface.json` — `{"protein_residues": [...], "np_nodes": [...], "top_fraction": ...}`.
- `predictions/pni/corona_ranking.csv` — columns `np_name,protein,score` (score = predicted propensity to form a complex).

## report/ (node N9)

- `report/report.md` — the reproduction report.
- `report/claims.yaml` — list of `{claim, verdict: supported|inconclusive|contradicted, evidence: [paths], notes}`.
- `report/toolcards/<step>.yaml` — draft tool cards for validated steps, in the format of the repository's `toolcards/`.
