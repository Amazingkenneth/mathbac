"""Apply the authors' saved GE+GT DNN fold models to their held-out features, as PPI_ML_prediction.py does."""
import sys, numpy as np, yaml
from pathlib import Path
from tensorflow import keras
sys.path.insert(0, "/home/user/mathbac")
from mathbac.science import pdb
C = Path("/srv/mathbac/capsule/data")
cols_all = ['c1','c1_pos','c1_neg','c1_polar','c1_amp','c1_hp','c1_hp_idx','c1_rd','c1_shell','c1_poc','c1_N_count','c1_C_count','c1_O_count','c1_H_count','c1_S_count','c1_N_charge','c1_C_charge','c1_O_charge','c1_H_charge','c1_S_charge','c1_ollivier','c1_forman','c1_gnm','fd1','more_fd1_1','more_fd1_2','more_fd1_3','more_fd1_4','G_os1_5','G_os1_7','G_os1_10','G_os1_15']
cols = cols_all + [c.replace('c1','c2').replace('fd1','fd2').replace('G_os1','G_os2') for c in cols_all] + ['distance']
gege = ['c1_rd','c1_shell','c1_poc','G_os1_5','G_os1_7','G_os1_10','G_os1_15','c1_ollivier','c1_forman','c1_gnm','fd1','more_fd1_1','more_fd1_2','more_fd1_3','more_fd1_4']
gege = gege + [c.replace('c1','c2').replace('fd1','fd2').replace('G_os1','G_os2') for c in gege]
idx = [cols.index(c) for c in gege]
models = [keras.models.load_model(str(C / f"dnn_models/re_model_GE_GT{k}.h5"), compile=False) for k in range(10)]
authors = yaml.safe_load(open("/home/user/mathbac/mathbac/eval/gold/cha2022/heldout_authors.yaml"))
for pid in ["6wzo", "6nqn", "1c29"]:
    T = np.load(C / f"processed_test_data/PPI/data_feature_{pid}_AB.npy", allow_pickle=True).astype(float)
    X = T[:, idx]
    # released code: forward-fill NaNs, unscaled features, mean of class-0 probability over the 10 models
    import pandas as pd
    X = pd.DataFrame(X).fillna(method="ffill").to_numpy()
    p0 = np.mean([m.predict(X, batch_size=8192)[:, 0] for m in models], axis=0)
    a = pdb.load_chain(pid, "A", Path("/home/user/mathbac/cache/pdb")); b = pdb.load_chain(pid, "B", Path("/home/user/mathbac/cache/pdb"))
    assert len(T) == len(a.resnums) * len(b.resnums), (len(T), len(a.resnums), len(b.resnums))
    n = int(len(p0) * 0.001)
    top = np.argsort(p0)[-n:]
    A = {a.resnums[i] for i in np.unique(top // len(b.resnums))}; B = {b.resnums[j] for j in np.unique(top % len(b.resnums))}
    pa, pb = set(authors[pid.upper()]["dnn_GE_GT_predicted"]["A"]), set(authors[pid.upper()]["dnn_GE_GT_predicted"]["B"])
    print(pid, f"predicted {len(A)}/{len(B)} residues; overlap with published Fig. 4 sets: A {len(A & pa)}/{len(pa)}, B {len(B & pb)}/{len(pb)}")
    # pair-level confusion matrix with argmax predictions (what the released code saves as *_cm_*.npy)
    y = (T[:, -1] == 0).astype(int)          # class 0 = contact
    pred = (p0 >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    acc = (tp + tn) / len(y)
    # SI-style guess: metrics with non-contact as the positive class
    p1 = tn / (tn + fn) if tn + fn else 0; r1 = tn / (tn + fp) if tn + fp else 0
    print(f"   pair-level: contacts {y.sum()}/{len(y)}, accuracy {acc:.3f}, contact precision {tp/max(tp+fp,1):.3f} recall {tp/max(tp+fn,1):.3f}; "
          f"non-contact precision {p1:.3f} recall {r1:.3f}")
