"""Helpers for acceptance checks: small, dependency-free readers for the artifact contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def require_columns(path: Path, required: list[str]) -> list[str]:
    if not Path(path).exists():
        return [f"{path} does not exist"]
    header, _ = read_csv(path)
    missing = [c for c in required if c not in header]
    return [f"{Path(path).name}: missing columns {missing}"] if missing else []


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(Path(path).read_text())


def load_npz(path: Path, keys: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    try:
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in keys if k not in z.files]
            return {k: z[k] for k in z.files}, ([f"{Path(path).name}: missing arrays {missing}"] if missing else [])
    except (OSError, ValueError) as e:
        return {}, [f"{Path(path).name}: cannot be read as npz without pickle ({e})"]


def chain_files(workspace: Path, split: str | None = None) -> list[tuple[str, str]]:
    """(pdb_id, chain) for every chain referenced in data/pairs.csv."""
    _, rows = read_csv(Path(workspace) / "data" / "pairs.csv")
    out = []
    for r in rows:
        if split and r.get("split") != split:
            continue
        for ch in (r["chain_a"], r["chain_b"]):
            key = (r["pdb_id"].lower(), ch)
            if key not in out:
                out.append(key)
    return out


def limit(problems: list[str], n: int = 15) -> list[str]:
    return problems if len(problems) <= n else problems[:n] + [f"... and {len(problems) - n} more problems"]
