#!/usr/bin/env bash
# N6a then N6b for the substitute-data run. Same settings as the L3 run (100 DNN epochs).
PY=/opt/mathbac/env/bin/python
$PY scripts/n6a_classical_cv.py --models lr,gnb,rf,xgb,svm > logs_n6a.txt 2>&1; echo "N6a EXIT $?" >> logs_n6a.txt
$PY scripts/n6a_classical_cv.py --models lr,gnb,svm --scale none --out ml/cv/predictions_unscaled > logs_n6a_unscaled.txt 2>&1; echo "EXIT $?" >> logs_n6a_unscaled.txt
for s in GE_GT GT all GE CH CH_GE CH_GT; do
  PATH=/opt/mathbac/env/bin:$PATH python /opt/mathbac/tools/dnn_cv.py --dataset ml/datasets/$s.npz --subset $s --out-dir ml --folds 10 --epochs 100 > logs_dnn_$s.txt 2>&1
  echo "subset $s exit $?" >> logs_n6b.txt
done
echo "N6 DONE" >> logs_n6b.txt
