"""Command-line training: 5-fold CV of the level-1 models, optional 2-level stack, and a test submission.

The building blocks live in src/stacking.py (also used by the notebook). Example (final model):
  python src/train.py --weighted --stack --extra outputs/grammar_features.csv outputs/deberta_large_s*.csv \
      outputs/whisper_ft_preds.csv --avg deberta_large=... --sub outputs/final.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import stacking as S

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

tr, te, HAND, PRED = S.load_table(a.transcripts, a.labels, a.extra, a.avg)
y = tr.label.values
w = S.batch_weights(tr, te)
fit_w = w if a.weighted else np.ones(len(tr))
fold = S.load_folds(tr, a.folds)
report = lambda name, pred: print(S.fmt(name, S.metrics(y, pred, tr.batch, w)))

report("mean predictor", np.full(len(y), y.mean()))
tr["tfidf_pred"], te["tfidf_pred"] = S.oof(S.tfidf_model, tr.text.values, y, fold, fit_w, te.text.values)
report("tfidf + ridge", tr.tfidf_pred.values)

FEATS = HAND + PRED
X, Xte = tr[FEATS].values, te[FEATS].values
report("features + ridge", S.oof(S.ridge_model, X, y, fold, fit_w)[0])

# single-level alternative: all features + all predictions -> gradient boosting / ridge, averaged
X2, X2te = tr[FEATS + ["tfidf_pred"]].values, te[FEATS + ["tfidf_pred"]].values
pred, pred_te = S.oof(S.hgb_model, X2, y, fold, fit_w, X2te)
report("features + tfidf -> HGB", pred)
pred_r, pred_r_te = S.oof(S.ridge_model, X2, y, fold, fit_w, X2te)
report("features + tfidf -> ridge", pred_r)
blend, blend_te = (pred + pred_r) / 2, (pred_te + pred_r_te) / 2
report("blend (HGB + ridge)", blend)
in_sample = (S.fit(S.hgb_model(), X2, y, fit_w).predict(X2) + S.fit(S.ridge_model(), X2, y, fit_w).predict(X2)) / 2
print(f"training RMSE (fit on all train, in-sample): {np.sqrt(np.mean((y - in_sample) ** 2)):.3f}")

if a.stack:
    # Level 1: hand-crafted features get their own model; every other component already is an OOF prediction.
    ph, ph_te = S.oof(S.hgb_model, tr[HAND].values, y, fold, fit_w, te[HAND].values)
    pr_, pr_te = S.oof(S.ridge_model, tr[HAND].values, y, fold, fit_w, te[HAND].values)
    comps = {"hand_features": ((ph + pr_) / 2, (ph_te + pr_te) / 2), "tfidf": (tr.tfidf_pred.values, te.tfidf_pred.values)}
    comps |= {c: (tr[c].values, te[c].values) for c in PRED}
    names = list(comps)
    P = np.column_stack([comps[n][0] for n in names])
    Pte = np.column_stack([comps[n][1] for n in names]) if len(te) else np.zeros((0, len(names)))
    for n in names:
        report(f"  component: {n}", comps[n][0])
    stack = S.nested_stack(P, y, fold, fit_w)
    report("STACK (nested CV)", stack)
    coef, b0 = S.nnls_weights(P, y, fit_w)
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
