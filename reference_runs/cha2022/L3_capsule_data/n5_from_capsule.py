"""N5 (L3): the authors' precomputed balanced training arrays, converted to the artifact contract.

Released arrays: features then a label column where class 0 = contact (<= 7 Å). Contract: y = 1 for contact.
Column layout is the released one (chain 1 descriptors, then chain 2); see manifest feature_names.
"""
import json
from pathlib import Path
import numpy as np
P = Path("materials/precomputed/processed_data_training")
FILES = {"all": ("binary_under_all7.npy", 64), "CH": ("CH_descriptor_7.npy", 34), "GE": ("GE_descriptor_7.npy", 14),
         "GT": ("GT_descriptor_7.npy", 16), "GE_GT": ("GE_GT_descriptor_7.npy", 30), "CH_GE": ("GE_CH_descriptor_7.npy", 48),
         "CH_GT": ("GT_CH_descriptor_7.npy", 50)}
out = Path("ml/datasets"); out.mkdir(parents=True, exist_ok=True)
manifest = {}
for name, (f, w) in FILES.items():
    a = np.load(P / f)
    X, lab = a[:, :w].astype(float), a[:, w]
    y = (lab == 0).astype(int)
    nan = int(np.isnan(X).sum())
    np.savez(out / f"{name}.npz", X=X, y=y, pair_index=np.array([f"capsule:{f}:{i}" for i in range(len(y))]))
    manifest[name] = {"n_rows": int(len(y)), "n_features": w, "feature_names": [f"released_col_{i}" for i in range(w)],
                      "balancing": "authors' undersampled arrays (binary_under / *_descriptor_7.npy)", "seed": None,
                      "source": f"capsule data/processed_data_training/{f}", "n_positive": int(y.sum()), "nan_values": nan}
    print(name, X.shape, "contacts", y.sum(), "NaN", nan)
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
