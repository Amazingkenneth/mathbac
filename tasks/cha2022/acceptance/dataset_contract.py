"""N3: pair list, per-chain Cα files and distance matrices are present and consistent."""

from pathlib import Path

import numpy as np

from mathbac.contracts import limit, load_json, load_npz, read_csv, require_columns

HELDOUT = {"6wzo", "6nqn", "1c29"}


def check(workspace: Path, node) -> list[str]:
    data = workspace / "data"
    problems = require_columns(data / "pairs.csv", ["pdb_id", "chain_a", "chain_b", "split"])
    if problems:
        return problems
    _, pairs = read_csv(data / "pairs.csv")
    splits = {r["split"] for r in pairs}
    if not splits <= {"train", "heldout"}:
        problems.append(f"pairs.csv: unknown split values {sorted(splits - {'train', 'heldout'})}")
    held = {r["pdb_id"].lower() for r in pairs if r["split"] == "heldout"}
    if not HELDOUT <= held:
        problems.append(f"pairs.csv: held-out complexes missing: {sorted(HELDOUT - held)}")
    train = {r["pdb_id"].lower() for r in pairs if r["split"] == "train"}
    if train & HELDOUT:
        problems.append(f"held-out complexes also appear in the training split: {sorted(train & HELDOUT)}")
    problems += require_columns(data / "substitutions.csv", ["requested_id", "used_id", "reason"])

    n_res: dict[tuple[str, str], int] = {}
    for r in pairs:
        for ch in (r["chain_a"], r["chain_b"]):
            key = (r["pdb_id"].lower(), ch)
            if key in n_res:
                continue
            f = data / "chains" / f"{key[0]}_{ch}.csv"
            cols = require_columns(f, ["resnum", "resname", "x", "y", "z"])
            if cols:
                problems += cols
                continue
            n_res[key] = len(read_csv(f)[1])
        pdb, a, b = r["pdb_id"].lower(), r["chain_a"], r["chain_b"]
        npz = data / "distances" / f"{pdb}_{a}{b}.npz"
        if not npz.exists():
            problems.append(f"missing distance matrix {npz.relative_to(workspace)}")
            continue
        arr, errs = load_npz(npz, ["D", "res_a", "res_b"])
        problems += errs
        if errs:
            continue
        exp = (n_res.get((pdb, a)), n_res.get((pdb, b)))
        if None not in exp and arr["D"].shape != exp:
            problems.append(f"{npz.name}: D has shape {arr['D'].shape}, chain files give {exp}")
        if arr["D"].size and (np.nanmin(arr["D"]) < 0 or not np.isfinite(arr["D"]).all()):
            problems.append(f"{npz.name}: distances must be finite and non-negative")

    try:
        summary = load_json(data / "dataset_summary.json")
        for k in ("n_pairs", "n_residue_pairs", "n_contacts"):
            if not isinstance(summary.get(k), int):
                problems.append(f"dataset_summary.json: {k} must be an integer")
    except Exception as e:
        problems.append(f"dataset_summary.json unreadable: {e}")
    return limit(problems)
