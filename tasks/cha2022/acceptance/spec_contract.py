"""N1: spec/spec.yaml has every required key with value, source and status."""

from pathlib import Path

from mathbac.contracts import load_yaml

REQUIRED = [
    "residue_representation", "contact_cutoff_A", "graph_edge_cutoff_A", "descriptors",
    "opd_neighbourhood_sizes", "gnm_modes_fraction", "node_curvature_aggregation", "mfd_features",
    "training_set", "class_balancing", "descriptor_subsets", "models", "dnn", "cv_folds", "metrics",
    "interface_threshold_top_fraction", "heldout_ppi", "np_coarse_graining", "np_descriptor_subset",
]
STATUSES = {"stated", "code_only", "unstated", "conflict"}


def check(workspace: Path, node) -> list[str]:
    try:
        spec = load_yaml(workspace / "spec" / "spec.yaml")
    except Exception as e:
        return [f"spec/spec.yaml is not valid YAML: {e}"]
    if not isinstance(spec, dict):
        return ["spec/spec.yaml must be a mapping"]
    problems = []
    for key in REQUIRED:
        entry = spec.get(key)
        if entry is None:
            problems.append(f"missing key {key}")
            continue
        if not isinstance(entry, dict) or "value" not in entry:
            problems.append(f"{key}: must be a mapping with value, source and status")
            continue
        if not entry.get("source"):
            problems.append(f"{key}: no source given")
        if entry.get("status") not in STATUSES:
            problems.append(f"{key}: status must be one of {sorted(STATUSES)}")
    desc = (spec.get("descriptors") or {}).get("value")
    if isinstance(desc, dict) and not all(k in desc for k in ("CH", "GE", "GT")):
        problems.append("descriptors.value must have CH, GE and GT lists")
    return problems
