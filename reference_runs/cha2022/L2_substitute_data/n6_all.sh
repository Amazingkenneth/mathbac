#!/usr/bin/env bash
# N6 for the substitute-data run: classical models with per-fold scaling, the linear models without scaling (the
# paper's condition; SVM uncalibrated, as Platt calibration on unscaled features is too slow), then the DNN at 100
# epochs. Then n6_metrics.py writes ml/cv/metrics.csv. The unscaled LR/GNB/SVM predictions are the Table 1 rows:
#   mkdir -p ml/cv/predictions_scaled; for m in lr gnb svm; do mv ml/cv/predictions/${m}_*.npz ml/cv/predictions_scaled/;
#   cp ml/cv/predictions_unscaled/${m}_*.npz ml/cv/predictions/; done
PY=/opt/mathbac/env/bin/python
$PY scripts/n6a_classical_cv.py --models lr,gnb,rf,xgb,svm > logs_n6a.txt 2>&1; echo "N6a EXIT $?" >> logs_n6a.txt
$PY scripts/n6a_classical_cv.py --models lr,gnb,svmd --scale none --out ml/cv/predictions_unscaled > logs_n6a_unscaled.txt 2>&1; echo "EXIT $?" >> logs_n6a_unscaled.txt
for s in GE_GT GT all GE CH CH_GE CH_GT; do
  PATH=/opt/mathbac/env/bin:$PATH python /opt/mathbac/tools/dnn_cv.py --dataset ml/datasets/$s.npz --subset $s --out-dir ml --folds 10 --epochs 100 > logs_dnn_$s.txt 2>&1
  echo "subset $s exit $?" >> logs_n6b.txt
done
echo "N6 DONE" >> logs_n6b.txt
