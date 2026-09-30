"""Model building blocks shared by src/train.py (command line) and the notebook.

Level 1: every component produces out-of-fold (OOF) predictions on the same 5 folds
         - hand-crafted fluency + grammar features -> gradient boosting & ridge (averaged)
         - TF-IDF of the transcript -> ridge
         - neural components (DeBERTa, LoRA-Whisper, ...) trained by their own GPU scripts, loaded as OOF CSVs
Level 2: a non-negative weighted average (+ intercept) of the level-1 predictions, fit with importance
         weights so the training batch mix matches the test batch mix; its CV score is nested.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import load as load_features

N_FOLDS = 5


# ---------------------------------------------------------------- data
def batch_of(duration):
    """Recording batch from clip length: ~45 s clips dominate the test set, ~60 s clips the training set."""
    return np.select([duration.between(44, 46), duration > 59], ["45s", "60s"], "other")


def load_table(transcripts, labels, extra=(), avg=()):
    """One row per clip: hand-crafted features + transcript + any extra per-clip CSVs.

    extra: CSVs keyed by filename+split. A CSV with a single value column is a model prediction; its column
           is named after the file (so seeds don't collide). avg: "name=stem1,stem2,..." seed-averages them.
    Returns train table (with label), test table, hand-feature columns, prediction columns.
    """
    feats, texts = load_features(transcripts)
    pred_cols = []
    for path in extra:
        e = pd.read_csv(path)
        cols = [c for c in e.columns if c not in ("filename", "split")]
        if len(cols) == 1:
            e = e.rename(columns={cols[0]: Path(path).stem})
            pred_cols.append(Path(path).stem)
        feats = feats.merge(e, on=["filename", "split"], how="left")
    for spec in avg:
        name, stems = spec.split("=")
        feats[name] = feats[stems.split(",")].mean(axis=1)
        feats = feats.drop(columns=stems.split(","))
        pred_cols = [c for c in pred_cols if c not in stems.split(",")] + [name]
    df = feats.merge(texts, on=["filename", "split"])
    lab = pd.read_csv(labels)
    # The 37 zero labels are exactly audio_5037..audio_5073, an appended block of ordinary speech (test ids only
    # go to 215): missing ratings stored as 0, not grammar scores. Training on them would bias every prediction down.
    lab = lab[lab.label > 0]
    tr = df[df.split == "train"].merge(lab, on="filename").reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    tr["batch"], te["batch"] = batch_of(tr.duration), batch_of(te.duration)
    all_cols = [c for c in feats.columns if c not in ("filename", "split")]
    med = tr[all_cols].median()  # clips with no scorable chunk get NaN grammar scores
    tr[all_cols], te[all_cols] = tr[all_cols].fillna(med), te[all_cols].fillna(med)
    hand = [c for c in all_cols if c not in pred_cols]
    return tr, te, hand, pred_cols


def batch_weights(tr, te):
    """Importance weights: (share of the clip's batch in test) / (share in train). Test is ~57 % 45 s clips vs ~17 %
    in train, so 45 s clips get ~3x weight. Used for the 'test-mix' CV score and for fitting."""
    if not len(te):
        return np.ones(len(tr))
    return tr.batch.map(te.batch.value_counts(normalize=True) / tr.batch.value_counts(normalize=True)).values


def load_folds(tr, path):
    """Fixed 5-fold split stratified on (batch, rounded score); created once and reused by every experiment."""
    path = Path(path)
    if path.exists():
        fold = tr[["filename"]].merge(pd.read_csv(path), on="filename", how="left").fold.values
    else:
        key = tr.batch + "_" + np.clip(np.round(tr.label.values), 1, 5).astype(int).astype(str)
        fold = np.zeros(len(tr), int)
        for k, (_, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=42).split(tr, key)):
            fold[va] = k
        pd.DataFrame({"filename": tr.filename, "fold": fold}).to_csv(path, index=False)
    assert not np.isnan(fold.astype(float)).any(), "folds file doesn't cover all training clips - delete it to rebuild"
    return fold


# ---------------------------------------------------------------- level-1 models
def tfidf_model():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), Ridge(alpha=3.0))


def ridge_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def hgb_model():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                         min_samples_leaf=20, l2_regularization=1.0, random_state=0)


def fit(model, X, y, w):
    """Fit a model or pipeline with sample weights routed to its final estimator."""
    key = f"{model.steps[-1][0]}__sample_weight" if hasattr(model, "steps") else "sample_weight"
    return model.fit(X, y, **{key: w})


def oof(make, X, y, fold, w, X_test=None):
    """Out-of-fold predictions (each clip predicted by a model that never saw it) + a full-train fit for test."""
    pred = np.zeros(len(y))
    for k in range(N_FOLDS):
        trn = fold != k
        pred[~trn] = fit(make(), X[trn], y[trn], w[trn]).predict(X[~trn])
    test = fit(make(), X, y, w).predict(X_test) if X_test is not None and len(X_test) else np.zeros(0)
    return pred, test


# ---------------------------------------------------------------- level 2
def nnls_weights(P, y, w):
    """Non-negative component weights + free intercept, least squares with sample weights w."""
    A = np.column_stack([P, np.ones(len(y)), -np.ones(len(y))]) * np.sqrt(w)[:, None]
    c = nnls(A, y * np.sqrt(w))[0]
    return c[:-2], c[-2] - c[-1]


def nested_stack(P, y, fold, w):
    """Level-2 CV predictions: the weights used for fold k are fit on the other folds only."""
    out = np.zeros(len(y))
    for k in range(N_FOLDS):
        coef, b0 = nnls_weights(P[fold != k], y[fold != k], w[fold != k])
        out[fold == k] = b0 + P[fold == k] @ coef
    return out


# ---------------------------------------------------------------- evaluation
def metrics(y, pred, batch, w):
    """RMSE / Pearson overall and on the 45 s batch, plus RMSE re-weighted to the test set's batch mix."""
    rmse = lambda m: float(np.sqrt(np.mean((y[m] - pred[m]) ** 2)))
    r = lambda m: float(pearsonr(y[m], pred[m])[0]) if np.ptp(pred[m]) > 1e-9 else float("nan")  # constant -> undefined
    s45 = np.asarray(batch) == "45s"
    return {"CV RMSE": rmse(slice(None)), "CV r": r(slice(None)), "45s RMSE": rmse(s45), "45s r": r(s45),
            "test-mix RMSE": float(np.sqrt(np.sum(w * (y - pred) ** 2) / w.sum()))}


def fmt(name, m):
    return (f"{name:28s} RMSE {m['CV RMSE']:.3f}  r {m['CV r']:.3f} | 45s: RMSE {m['45s RMSE']:.3f} r {m['45s r']:.3f}"
            f" | test-mix RMSE {m['test-mix RMSE']:.3f}")
