#!/usr/bin/env bash
# N4b/c driver: fast stage (cached per chain), then ORC/FRC, then write. Resumable.
export PAIRS=data/pairs_with_reference.csv
PY=/opt/mathbac/env/bin/python
$PY scripts/n4_descriptors.py --stage fast --procs 3 > logs_n4_fast.txt 2>&1; echo "FAST EXIT $?" >> logs_n4_fast.txt
$PY scripts/n4_descriptors.py --stage orc > logs_n4_orc.txt 2>&1; echo "ORC EXIT $?" >> logs_n4_orc.txt
$PY scripts/n4_descriptors.py --stage write > logs_n4_write.txt 2>&1; echo "WRITE EXIT $?" >> logs_n4_write.txt
echo "N4 DONE" >> logs_n4_write.txt
