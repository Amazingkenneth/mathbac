"""Minimal PDB reader for the evaluator: Cα coordinates per chain, first model, first altloc.

The evaluator computes ground truth from RCSB coordinates itself instead of trusting
files the agents wrote.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RCSB = "https://files.rcsb.org/download/{id}.pdb"


@dataclass
class Chain:
    resnums: list[str]   # resSeq + insertion code, e.g. "52", "52A"
    resnames: list[str]
    ca: np.ndarray       # (n, 3)

    def label(self, i: int) -> str:
        return f"{self.resnames[i]}{self.resnums[i]}"


def fetch_pdb(pdb_id: str, cache_dir: Path) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{pdb_id.upper()}.pdb"
    if not path.exists():
        with urllib.request.urlopen(RCSB.format(id=pdb_id.upper()), timeout=60) as r:
            path.write_bytes(r.read())
    return path


def parse_ca(text: str, chain_id: str) -> Chain:
    resnums, resnames, coords = [], [], []
    seen: set[str] = set()
    for line in text.splitlines():
        rec = line[:6]
        if rec.startswith("ENDMDL"):
            break
        if rec != "ATOM  ":  # protein residues only, as in the released code (PandasPdb df['ATOM'])
            continue
        if line[12:16].strip() != "CA" or line[21] != chain_id:
            continue
        altloc = line[16]
        if altloc not in (" ", "A"):
            continue
        rn = line[22:26].strip() + line[26].strip()
        if rn in seen:
            continue
        seen.add(rn)
        resnums.append(rn)
        resnames.append(line[17:20].strip())
        coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if not coords:
        raise ValueError(f"no Cα atoms found for chain {chain_id}")
    return Chain(resnums, resnames, np.array(coords, dtype=float))


def load_chain(pdb_id: str, chain_id: str, cache_dir: Path) -> Chain:
    return parse_ca(fetch_pdb(pdb_id, cache_dir).read_text(), chain_id)


def distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


def interface(a: Chain, b: Chain, cutoff: float = 7.0) -> tuple[set[str], set[str]]:
    d = distance_matrix(a.ca, b.ca) < cutoff
    ia = {a.resnums[i] for i in np.where(d.any(axis=1))[0]}
    ib = {b.resnums[j] for j in np.where(d.any(axis=0))[0]}
    return ia, ib
