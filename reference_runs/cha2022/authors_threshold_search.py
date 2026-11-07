"""Which threshold / scaling reproduces the published Fig. 4 interface sets from the authors' own models and features?"""
import sys, numpy as np, yaml, pandas as pd
from pathlib import Path
from tensorflow import keras
sys.path.insert(0, "/home/user/mathbac"); sys.path.insert(0, "/srv/mathbac/checks")
from mathbac.science import pdb
from authors_models_heldout import cols, idx, C  # noqa
models = {s: [keras.models.load_model(str(C / f"dnn_models/re_model_{s}{k}.h5"), compile=False) for k in range(10)] for s in ["GE_GT"]}
train = np.load(C / "processed_data_training/GE_GT_descriptor_7.npy")[:, :30]
lo, hi = train.min(0), train.max(0); rng = np.where(hi > lo, hi - lo, 1)
authors = yaml.safe_load(open("/home/user/mathbac/mathbac/eval/gold/cha2022/heldout_authors.yaml"))
for pid in ["6wzo", "6nqn", "1c29"]:
    T = np.load(C / f"processed_test_data/PPI/data_feature_{pid}_AB.npy", allow_pickle=True).astype(float)
    X = pd.DataFrame(T[:, idx]).fillna(method="ffill").to_numpy()
    a = pdb.load_chain(pid, "A", Path("/home/user/mathbac/cache/pdb")); b = pdb.load_chain(pid, "B", Path("/home/user/mathbac/cache/pdb"))
    pa, pb = set(authors[pid.upper()]["dnn_GE_GT_predicted"]["A"]), set(authors[pid.upper()]["dnn_GE_GT_predicted"]["B"])
    pub = {("A", x) for x in pa} | {("B", x) for x in pb}
    best = []
    for scale in ["none", "minmax"]:
        Xs = X if scale == "none" else (X - lo) / rng
        for col in [0, 1]:
            p = np.mean([m.predict(Xs, batch_size=8192)[:, col] for m in models["GE_GT"]], axis=0)
            for frac in [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]:
                n = max(int(len(p) * frac), 1); top = np.argsort(p)[-n:]
                ours = {("A", a.resnums[i]) for i in np.unique(top // len(b.resnums))} | {("B", b.resnums[j]) for j in np.unique(top % len(b.resnums))}
                jac = len(ours & pub) / len(ours | pub)
                best.append((round(jac, 3), scale, f"class{col}", frac, len(ours)))
    best.sort(reverse=True)
    print(pid, "published", len(pub), "best:", best[:4])
