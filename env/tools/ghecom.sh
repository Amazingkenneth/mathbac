#!/usr/bin/env bash
# Usage: ghecom.sh <pdb_path> <chain> <out_txt>
# Runs GHECOM on one protein chain (ATOM records only, first altloc) and writes the
# residue-based table that the released descriptor code reads (43 header rows;
# columns 3, 4, 7 = shell volume, R_inacc, pocketness).
# The flags below follow GHECOM's residue-output mode; the smoke test checks them on 6WZO.
set -euo pipefail
pdb="$1"; chain="$2"; out="$3"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
awk -v c="$chain" '/^ATOM/ && substr($0,22,1)==c && (substr($0,17,1)==" " || substr($0,17,1)=="A")' "$pdb" > "$tmp/chain.pdb"
echo "END" >> "$tmp/chain.pdb"
mkdir -p "$(dirname "$out")"
ghecom -M M -ipdb "$tmp/chain.pdb" -ores "$out"
n=$(grep -c '^ATOM\|^HETATM' "$tmp/chain.pdb" || true)
echo "ghecom: chain $chain, $n atoms -> $out"
