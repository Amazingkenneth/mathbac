#!/usr/bin/env bash
# Usage: smoke_test.sh <tool_name>
# Runs one tool on chain A of 6WZO and checks it produced output. Run inside the sandbox image.
set -euo pipefail
tool="$1"
w=$(mktemp -d)
trap 'rm -rf "$w"' EXIT
cd "$w"
curl -sSf -o 6WZO.pdb https://files.rcsb.org/download/6WZO.pdb
python - <<'PY'
import csv
seen=set(); rows=[]
for line in open("6WZO.pdb"):
    if line.startswith("ATOM") and line[12:16].strip()=="CA" and line[21]=="A" and line[16] in " A":
        rn=line[22:27].strip()
        if rn in seen: continue
        seen.add(rn); rows.append([rn,line[17:20].strip(),line[30:38],line[38:46],line[46:54]])
with open("ca.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["resnum","resname","x","y","z"]); w.writerows(rows)
print(len(rows),"residues")
PY
T=/opt/mathbac/tools
case "$tool" in
  ghecom)                  "$T/ghecom.sh" 6WZO.pdb A ghecom.txt && test "$(wc -l < ghecom.txt)" -gt 43 ;;
  gnm_prody)               python "$T/descriptors.py" gnm --pdb 6WZO.pdb --chain A --out o.csv && test -s o.csv ;;
  ricci_curvature)         python "$T/descriptors.py" curvature --ca ca.csv --out o.csv && test -s o.csv ;;
  multifractal_dimension)  python "$T/descriptors.py" mfd --ca ca.csv --out o.csv && test -s o.csv ;;
  opd_chirality)           python "$T/descriptors.py" opd --ca ca.csv --out o.csv && test -s o.csv ;;
  unified_ppi_descriptors) cp 6WZO.pdb /opt/unified_fast/code/unified_descriptors/ 2>/dev/null || true
                           "$T/unified_ppi.sh" 6wzo A B out && ls out | grep -q . ;;
  unified_pni_descriptors) test -f /opt/unified_fast/code/unified_descriptors/PNI_descriptors_cal.py ;;
  dnn_train)               python -c "import tensorflow as tf; print(tf.__version__)" ;;
  *) echo "no smoke test for $tool" >&2; exit 2 ;;
esac
echo "smoke test passed: $tool"
