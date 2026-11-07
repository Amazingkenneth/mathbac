#!/usr/bin/env bash
# N6b: tenfold CV of the paper's DNN on every subset with the released training loop (tools/dnn_cv.py).
# Deviation: 100 epochs instead of 1000 (CPU only; ~1.1 s per epoch per fold). Best epoch by test-fold accuracy, as released.
set -uo pipefail
for s in GE_GT GT all GE CH CH_GE CH_GT; do
  python /opt/mathbac/tools/dnn_cv.py --dataset ml/datasets/$s.npz --subset $s --out-dir ml --folds 10 --epochs 100 > logs_dnn_$s.txt 2>&1
  echo "subset $s exit $?"
done
echo "N6b DONE"
