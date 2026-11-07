"""k-fold cross-validation of the paper's DNN, mirroring DNN_roc_auc_cross_val in the released DNN_train.py.

    python dnn_cv.py --dataset ml/datasets/GT.npz --subset GT --out-dir ml [--folds 10] [--epochs 1000]

Writes ml/cv/predictions/dnn_<subset>.npz (fold, y_true, y_score with 1 = contact) and saves
fold models under ml/models/dnn_<subset>/.

Released-code behaviour reproduced here, with switches where it is questionable:
- MinMax scaling fitted on the full X before splitting (--scale-per-fold to fit on train folds only).
- KFold(shuffle=True, random_state=0).
- BatchNorm, then 3 x Dense(512, relu) + Dropout(0.5), then Dense(2, softmax).
- SparseCategoricalCrossentropy(from_logits=True) on a softmax output (--fix-logits to use from_logits=False).
- Adam, batch 128, up to 1000 epochs, keeping the epoch with the best accuracy on the *test fold*
  (--val-from-train to select on a split of the training folds instead).
- The released code then reloads a fixed path "best_model" instead of the fold checkpoint; here we
  reload the fold checkpoint.
- In the released arrays label 0 is the contact class; ml/datasets uses 1 = contact, so no flip is needed.
"""

import argparse
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--subset", required=True)
    p.add_argument("--out-dir", default="ml")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--scale-per-fold", action="store_true")
    p.add_argument("--fix-logits", action="store_true")
    p.add_argument("--val-from-train", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import tensorflow as tf
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(a.seed)
    z = np.load(a.dataset)
    X_raw, y = z["X"].astype("float32"), z["y"].astype(int)
    X_all = X_raw if a.scale_per_fold else MinMaxScaler().fit_transform(X_raw)
    model_dir = os.path.join(a.out_dir, "models", f"dnn_{a.subset}")
    os.makedirs(model_dir, exist_ok=True)
    folds, ys, scores = [], [], []

    cv = KFold(n_splits=a.folds, shuffle=True, random_state=0)
    for k, (tr, te) in enumerate(cv.split(X_all, y)):
        if a.scale_per_fold:
            sc = MinMaxScaler().fit(X_raw[tr])
            X_tr, X_te = sc.transform(X_raw[tr]), sc.transform(X_raw[te])
        else:
            X_tr, X_te = X_all[tr], X_all[te]
        y_tr, y_te = y[tr], y[te]
        if a.val_from_train:
            X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.1, random_state=a.seed)
        else:
            X_val, y_val = X_te, y_te
        model = keras.Sequential([
            layers.Input(X_tr.shape[1:]), layers.Flatten(), layers.BatchNormalization(),
            layers.Dense(512, activation="relu"), layers.Dropout(0.5),
            layers.Dense(512, activation="relu"), layers.Dropout(0.5),
            layers.Dense(512, activation="relu"), layers.Dropout(0.5),
            layers.Dense(2, activation="softmax"),
        ])
        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=not a.fix_logits),
                      optimizer=keras.optimizers.Adam(), metrics=["accuracy"])
        ckpt = os.path.join(model_dir, f"fold{k}.h5")
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val), batch_size=a.batch_size, epochs=a.epochs, shuffle=True,
                  verbose=2, callbacks=[keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True, mode="max")])
        best = keras.models.load_model(ckpt)
        prob = best.predict(X_te)[:, 1]
        folds.append(np.full(len(te), k)); ys.append(y_te); scores.append(prob)
        print(f"fold {k}: {len(te)} rows")

    out = os.path.join(a.out_dir, "cv", "predictions", f"dnn_{a.subset}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, fold=np.concatenate(folds), y_true=np.concatenate(ys), y_score=np.concatenate(scores))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
