# Reference runs for Cha et al. 2022

Stage scripts from two direct runs of the reproduction (one Claude agent acting as every worker role, because no model
API key was available for the multi-agent harness). They follow the reference task graph and write the artifact
contracts in `tasks/cha2022/contracts/artifacts.md`, so `mathbac eval <run_dir>` scores them like an agent run.

| Folder | Access level | Data |
|---|---|---|
| `L2_substitute_data/` | L2 (paper, SI, released code) | 200 X-ray heterodimers curated from RCSB (`n3_curate.py`), because the paper's pair list is only in the Code Ocean capsule; CH charges recomputed with PDB2PQR/CHARMM; GHECOM descriptors approximated by multiscale morphology |
| `L3_capsule_data/` | L3 (+ capsule data) | the capsule's pair list, structures, GHECOM outputs, CH tables, balanced training arrays and precomputed test features |

Two diagnostics use the authors' own saved models and test features:

- `authors_models_heldout.py`: runs the capsule's GE+GT DNN models on the capsule's held-out features exactly as
  `PPI_ML_prediction.py` does.
- `authors_threshold_search.py`: searches threshold, scaling and output column for settings that reproduce the
  published Fig. 4 interface sets. None does.

Results and the list of inconsistencies found are in each run's `report/report.md` and in `tasks/cha2022/README.md`.
Run the scripts from a run workspace (they use relative paths `data/`, `features/`, `ml/`, `predictions/`), with the
capsule environment from `env/setup_local.sh`.
