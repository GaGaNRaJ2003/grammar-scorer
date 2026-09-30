"""Audio-only model: ridge regression on frozen WavLM embeddings (from src/audio_embed.py).

The layer choice (mean of layers 6-9) is fixed up front, not selected on the CV folds, so the OOF
score stays honest; the per-layer table is printed only as analysis for the notebook.
Ridge's alpha is chosen inside each training fold (RidgeCV's efficient leave-one-out).
Output: outputs/wavlm_preds.csv (OOF for train, full-train fit for test) -> feature for src/train.py.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

p = argparse.ArgumentParser()
p.add_argument("--emb", default="outputs/wavlm_emb.npz")
p.add_argument("--labels", default="data/train.csv")
p.add_argument("--folds", default="outputs/folds.csv")
p.add_argument("--layers", default="6,7,8,9")
p.add_argument("--out", default="outputs/wavlm_preds.csv")
a = p.parse_args()

z = np.load(a.emb)
names = pd.Series(z["names"]).str.split("/", expand=True).set_axis(["split", "filename"], axis=1)
lab = pd.read_csv(a.labels)
lab = lab[lab.label > 0]  # audio_5037..5073 zeros are missing ratings (see src/train.py)
tr_idx = names.reset_index().merge(lab, on="filename").merge(pd.read_csv(a.folds), on="filename")
tr_idx = tr_idx[tr_idx.split == "train"]
te_idx = names[names.split == "test"]
y, fold = tr_idx.label.values, tr_idx.fold.values
make = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))


def oof(X):
    pred = np.zeros(len(y))
    for k in np.unique(fold):
        pred[fold == k] = make().fit(X[fold != k], y[fold != k]).predict(X[fold == k])
    return pred


E = z["emb"]
Xtr = E[tr_idx["index"].values]
for layer in range(E.shape[1]):  # analysis only
    pr = oof(Xtr[:, layer])
    print(f"layer {layer:2d}: RMSE {np.sqrt(np.mean((pr - y) ** 2)):.3f}  r {np.corrcoef(pr, y)[0, 1]:.3f}")

L = [int(l) for l in a.layers.split(",")]
X, Xte = Xtr[:, L].mean(1), E[te_idx.index.values][:, L].mean(1)
pr = oof(X)
print(f"layers {L} (used): OOF RMSE {np.sqrt(np.mean((pr - y) ** 2)):.3f}  r {np.corrcoef(pr, y)[0, 1]:.3f}")
pd.concat([pd.DataFrame({"filename": tr_idx.filename, "split": "train", "wavlm_pred": pr}),
           pd.DataFrame({"filename": te_idx.filename, "split": "test", "wavlm_pred": make().fit(X, y).predict(Xte)})]
          ).to_csv(a.out, index=False)
print("wrote", a.out)
