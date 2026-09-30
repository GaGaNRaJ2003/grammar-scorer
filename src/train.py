"""5-fold CV of the baseline models and a test submission.

Folds are stratified on (recording batch, label) and saved, so every experiment uses the same splits.
Metrics are reported overall and on the ~45 s batch, which dominates the test set.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import load

p = argparse.ArgumentParser()
p.add_argument("--transcripts", default="outputs/transcripts_fw_large_v3.jsonl")
p.add_argument("--labels", default="data/train.csv")
p.add_argument("--sub", default="outputs/submission.csv")
p.add_argument("--folds", default="outputs/folds.csv")
p.add_argument("--extra", nargs="*", default=[],
               help="per-clip CSVs keyed by filename+split: grammar_features.csv, deberta_preds.csv (OOF), ...")
p.add_argument("--weighted", action="store_true",
               help="fit with importance weights so the training batch mix matches the test batch mix")
p.add_argument("--avg", nargs="*", default=[],
               help="seed-average prediction files into one component: name=file_stem1,file_stem2")
p.add_argument("--stack", action="store_true",
               help="2-level model: each component predicts alone, then a non-negative weighted average")
a = p.parse_args()

feats, texts = load(a.transcripts)
PRED = []  # model-prediction columns (vs hand-crafted features)
for path in a.extra:
    e = pd.read_csv(path)
    cols = [c for c in e.columns if c not in ("filename", "split")]
    if len(cols) == 1:  # a prediction file: name its column after the file so seeds don't collide
        e = e.rename(columns={cols[0]: Path(path).stem})
        PRED.append(Path(path).stem)
    feats = feats.merge(e, on=["filename", "split"], how="left")
for spec in a.avg:
    name, stems = spec.split("=")
    feats[name] = feats[stems.split(",")].mean(axis=1)
    feats = feats.drop(columns=stems.split(","))
    PRED = [c for c in PRED if c not in stems.split(",")] + [name]
df = feats.merge(texts, on=["filename", "split"])
lab = pd.read_csv(a.labels)
# The 37 zero labels are exactly audio_5037..audio_5073, an appended block of ordinary speech (test ids only
# go to 215): missing ratings stored as 0, not grammar scores. Training on them would bias every prediction down.
lab = lab[lab.label > 0]
tr = df[df.split == "train"].merge(lab, on="filename").reset_index(drop=True)
te = df[df.split == "test"].reset_index(drop=True)
y = tr.label.values
batch = lambda d: np.select([d.duration.between(44, 46), d.duration > 59], ["45s", "60s"], "other")
tr["batch"], te["batch"] = batch(tr), batch(te)
# Importance weights for the batch shift: test is ~57% 45 s clips vs ~17% in train. Used for the
# "test-mix" CV score (a better leaderboard estimate) and, with --weighted, for fitting.
w = (tr.batch.map(te.batch.value_counts(normalize=True) / tr.batch.value_counts(normalize=True)).values
     if len(te) else np.ones(len(tr)))
fit_w = w if a.weighted else np.ones(len(tr))
FEATS = [c for c in feats.columns if c not in ("filename", "split")]
med = tr[FEATS].median()  # clips with no scorable chunk get NaN grammar scores
tr[FEATS], te[FEATS] = tr[FEATS].fillna(med), te[FEATS].fillna(med)

folds_path = Path(a.folds)
if folds_path.exists():
    fold = tr[["filename"]].merge(pd.read_csv(folds_path), on="filename", how="left").fold.values
else:
    key = tr.batch + "_" + np.clip(np.round(y), 1, 5).astype(int).astype(str)
    fold = np.zeros(len(tr), int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, key)):
        fold[va] = k
    pd.DataFrame({"filename": tr.filename, "fold": fold}).to_csv(folds_path, index=False)
assert not np.isnan(fold.astype(float)).any(), "folds.csv doesn't cover all training clips - delete it to rebuild"


def fit(m, X, yy, ww):
    """Fit a model or pipeline with sample weights routed to its final estimator."""
    key = f"{m.steps[-1][0]}__sample_weight" if hasattr(m, "steps") else "sample_weight"
    return m.fit(X, yy, **{key: ww})


def oof(make, X, Xte=None):
    """Out-of-fold predictions on train (+ a full-train fit for test)."""
    pred = np.zeros(len(y))
    for k in range(5):
        tr_k = fold != k
        pred[~tr_k] = fit(make(), X[tr_k], y[tr_k], fit_w[tr_k]).predict(X[~tr_k])
    return pred, (fit(make(), X, y, fit_w).predict(Xte) if Xte is not None and len(Xte) else np.zeros(0))


def report(name, pred):
    rm = lambda m: np.sqrt(np.mean((y[m] - pred[m]) ** 2))
    pr = lambda m: pearsonr(y[m], pred[m])[0] if pred[m].std() > 0 else float("nan")
    s45 = tr.batch.values == "45s"
    mix = np.sqrt(np.sum(w * (y - pred) ** 2) / w.sum())
    print(f"{name:28s} RMSE {rm(slice(None)):.3f}  r {pr(slice(None)):.3f} | 45s: RMSE {rm(s45):.3f} r {pr(s45):.3f}"
          f" | test-mix RMSE {mix:.3f}")


report("mean predictor", np.full(len(y), y.mean()))

tfidf = lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), Ridge(alpha=3.0))
tr["tfidf_pred"], te["tfidf_pred"] = oof(tfidf, tr.text.values, te.text.values)
report("tfidf + ridge", tr.tfidf_pred.values)

X, Xte = tr[FEATS].values, te[FEATS].values
ridge = lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))
report("features + ridge", oof(ridge, X)[0])

# ponytail: tfidf OOF preds feed the 2nd-level model with the same folds (standard stacking, tiny optimism)
X2, X2te = tr[FEATS + ["tfidf_pred"]].values, te[FEATS + ["tfidf_pred"]].values
hgb = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                            min_samples_leaf=20, l2_regularization=1.0, random_state=0)
pred, pred_te = oof(hgb, X2, X2te)
report("features + tfidf -> HGB", pred)
pred_r, pred_r_te = oof(ridge, X2, X2te)
report("features + tfidf -> ridge", pred_r)
blend, blend_te = (pred + pred_r) / 2, (pred_te + pred_r_te) / 2
report("blend (HGB + ridge)", blend)

train_rmse = np.sqrt(np.mean((y - (fit(hgb(), X2, y, fit_w).predict(X2) + fit(ridge(), X2, y, fit_w).predict(X2)) / 2) ** 2))
print(f"training RMSE (fit on all train, in-sample): {train_rmse:.3f}")

if a.stack:
    # Level 1: hand-crafted features get their own model; every other component already is an OOF prediction.
    HAND = [c for c in FEATS if c not in PRED]
    Xh, Xhte = tr[HAND].values, te[HAND].values
    ph, ph_te = oof(hgb, Xh, Xhte)
    pr_, pr_te = oof(ridge, Xh, Xhte)
    comps = {"hand_features": ((ph + pr_) / 2, (ph_te + pr_te) / 2), "tfidf": (tr.tfidf_pred.values, te.tfidf_pred.values)}
    comps |= {c: (tr[c].values, te[c].values) for c in PRED}
    names = list(comps)
    P = np.column_stack([comps[n][0] for n in names])
    Pte = np.column_stack([comps[n][1] for n in names]) if len(te) else np.zeros((0, len(names)))
    for n in names:
        report(f"  component: {n}", comps[n][0])

    def l2(Ptr, ytr, wtr):
        """Non-negative weights + free intercept, fit with the importance weights (weighted NNLS)."""
        from scipy.optimize import nnls
        A = np.column_stack([Ptr, np.ones(len(ytr)), -np.ones(len(ytr))]) * np.sqrt(wtr)[:, None]
        c = nnls(A, ytr * np.sqrt(wtr))[0]
        return c[:-2], c[-2] - c[-1]

    stack = np.zeros(len(y))  # nested: level-2 weights for fold k are fit on the other folds only
    for k in range(5):
        coef, b0 = l2(P[fold != k], y[fold != k], fit_w[fold != k])
        stack[fold == k] = b0 + P[fold == k] @ coef
    report("STACK (nested CV)", stack)
    coef, b0 = l2(P, y, fit_w)
    print("level-2 weights:", {n: round(float(c), 3) for n, c in zip(names, coef)}, "intercept", round(float(b0), 3))
    blend, blend_te = stack, b0 + Pte @ coef
    print(f"training RMSE, stack (in-sample): {np.sqrt(np.mean((y - (b0 + P @ coef)) ** 2)):.3f}")

pd.DataFrame({"filename": tr.filename, "batch": tr.batch, "label": y, "w": w, "oof": blend}).to_csv(
    Path(a.sub).with_suffix(".oof.csv"), index=False)  # for calibration analysis / notebook plots
if len(te):
    pd.DataFrame({"filename": te.filename, "batch": te.batch, "pred": blend_te}).to_csv(
        Path(a.sub).with_suffix(".test.csv"), index=False)
    sub = pd.DataFrame({"filename": te.filename, "label": np.clip(blend_te, 0, 5)})
    sub.to_csv(a.sub, index=False)
    print(f"wrote {a.sub}: {len(sub)} rows, mean {sub.label.mean():.2f}, std {sub.label.std():.2f}")
