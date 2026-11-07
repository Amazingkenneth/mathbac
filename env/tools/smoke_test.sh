#!/usr/bin/env bash
# Usage: smoke_test.sh <tool_name>
# Runs one tool on chain A of 6WZO (from RCSB) and checks it produced output. Exit 0 only on success.
set -uo pipefail
tool="$1"
w=$(mktemp -d)
trap 'rm -rf "$w"' EXIT
cd "$w"
fail() { echo "smoke test FAILED: $tool: $*"; exit 1; }
curl -sSf -o 6WZO.pdb https://files.rcsb.org/download/6WZO.pdb || fail "could not download 6WZO"
python - <<'PY' || fail "could not extract Cα coordinates"
import csv
seen=set(); rows=[]
for line in open("6WZO.pdb"):
    if line.startswith("ATOM") and line[12:16].strip()=="CA" and line[21]=="A" and line[16] in " A":
        rn=line[22:27].strip()
        if rn in seen: continue
        seen.add(rn); rows.append([rn,line[17:20].strip(),line[30:38],line[38:46],line[46:54]])
with open("ca.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["resnum","resname","x","y","z"]); w.writerows(rows)
PY
T=/opt/mathbac/tools
case "$tool" in
  ghecom)                  command -v ghecom >/dev/null || fail "ghecom binary not installed (source needs registration at pdbj.org/ghecom)"
                           "$T/ghecom.sh" 6WZO.pdb A ghecom.txt || fail "ghecom.sh failed"
                           [ "$(wc -l < ghecom.txt)" -gt 43 ] || fail "ghecom output too short" ;;
  gnm_prody)               python "$T/descriptors.py" gnm --pdb 6WZO.pdb --chain A --out o.csv || fail "gnm failed" ;;
  ricci_curvature)         python "$T/descriptors.py" curvature --ca ca.csv --out o.csv || fail "curvature failed" ;;
  multifractal_dimension)  python "$T/descriptors.py" mfd --ca ca.csv --out o.csv || fail "mfd failed" ;;
  opd_chirality)           python "$T/descriptors.py" opd --ca ca.csv --out o.csv || fail "opd failed" ;;
  unified_ppi_descriptors) python -c "import pymol" 2>/dev/null || fail "PyMOL not installed (PPI_descriptors_cal.py imports it)"
                           "$T/unified_ppi.sh" 6wzo A B out || fail "pipeline failed" ;;
  unified_pni_descriptors) python -c "import pymol" 2>/dev/null || fail "PyMOL not installed (PNI_descriptors_cal.py imports it)" ;;
  dnn_train)               python -c "import tensorflow as tf, sklearn; assert tf.__version__.startswith('2.3')" || fail "TensorFlow 2.3 missing" ;;
  *) fail "no smoke test defined" ;;
esac
case "$tool" in
  gnm_prody|ricci_curvature|multifractal_dimension|opd_chirality) [ -s o.csv ] || fail "no output written" ;;
esac
echo "smoke test passed: $tool"
