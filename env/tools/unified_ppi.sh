#!/usr/bin/env bash
# Usage: unified_ppi.sh <pdb_id> <chain_a> <chain_b> <out_dir>
# Runs the released PPI descriptor pipeline (unified_fast PPI_descriptors_cal.py) for one complex.
# The script imports a package named `descriptor_calc` that the repository does not contain;
# we expose unified_descriptors/ under that name.
set -euo pipefail
pdb="$1"; a="$2"; b="$3"; out="$4"
src=/opt/unified_fast/code/unified_descriptors
shim=$(mktemp -d)
trap 'rm -rf "$shim"' EXIT
ln -s "$src" "$shim/descriptor_calc"
mkdir -p "$out"
cd "$src"
PYTHONPATH="$shim:$src:${PYTHONPATH:-}" python PPI_descriptors_cal.py "$pdb" "$a" "$b" "$out"
