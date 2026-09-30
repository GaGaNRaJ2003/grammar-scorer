"""Fine-tune a pretrained text encoder (DeBERTa-v3) to regress the grammar score from the transcript.

Uses the saved CV folds: each fold's model predicts its held-out clips (out-of-fold, OOF) and the
test set; test predictions are averaged over folds. The number of epochs is fixed up front (not picked
on the validation fold) so the OOF score is an honest estimate. Output feeds src/train.py as a feature.
"""
import argparse, json, math, sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

p = argparse.ArgumentParser()
p.add_argument("--transcripts", default="outputs/transcripts_fw_large_v3.merged.jsonl")
p.add_argument("--labels", default="data/Dataset_Final/train.csv")
p.add_argument("--folds", default="outputs/folds.csv")
p.add_argument("--model", default="microsoft/deberta-v3-base")
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--max-len", type=int, default=256)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="outputs/deberta_preds.csv")
p.add_argument("--pauses", action="store_true",
               help="rebuild the text from word timings with (pause)/(long pause) markers, so the text model "
                    "sees hesitations that Whisper's clean transcript hides")
a = p.parse_args()


def with_pauses(words):
    """Silent-pause threshold 250 ms (de Jong & Bosker 2013, standard in fluency research); long pause > 1 s."""
    out, prev_end = [], None
    for w, start, end, *_ in words:
        gap = start - prev_end if prev_end is not None else 0.0
        if gap > 1.0:
            out.append(" (long pause)")
        elif gap > 0.25:
            out.append(" (pause)")
        out.append(w)
        prev_end = end
    return "".join(out).strip()

free, total = torch.cuda.mem_get_info()  # Guard: never share a GPU slice with someone else's job.
if total - free > 2 * 1024**3:
    sys.exit(f"GPU slice already in use ({(total - free) / 1024**3:.1f} GB used) - exiting")

rows = pd.DataFrame([json.loads(l) for l in open(a.transcripts, encoding="utf-8")])
if a.pauses:
    rows["text"] = rows.words.apply(with_pauses)
rows = rows[["filename", "split", "text"]]
lab = pd.read_csv(a.labels)
lab = lab[lab.label > 0]  # audio_5037..5073 zeros are missing ratings (see src/train.py)
tr = rows[rows.split == "train"].merge(lab, on="filename").merge(pd.read_csv(a.folds), on="filename")
te = rows[rows.split == "test"].reset_index(drop=True)
mu, sd = tr.label.mean(), tr.label.std()  # regress the standardised score: stabler with a fresh head

tok = AutoTokenizer.from_pretrained(a.model)


def loader(texts, ys=None, shuffle=False):
    enc = tok(list(texts), truncation=True, max_length=a.max_len)
    items = [{"input_ids": i, "attention_mask": m, **({"labels": float(y)} if ys is not None else {})}
             for i, m, y in zip(enc["input_ids"], enc["attention_mask"], ys if ys is not None else [0] * len(texts))]
    pad = lambda b: tok.pad(b, return_tensors="pt")
    return DataLoader(items, batch_size=a.bs, shuffle=shuffle, collate_fn=pad)


@torch.no_grad()
def predict(model, texts):
    model.eval()
    out = []
    for b in loader(texts):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(**{k: v.cuda() for k, v in b.items()}).logits.float().squeeze(-1).cpu())
    return torch.cat(out).numpy() * sd + mu


oof, test_pred = np.zeros(len(tr)), np.zeros(len(te))
for k in sorted(tr.fold.unique()):
    torch.manual_seed(a.seed + k)
    trn, val = tr[tr.fold != k], tr[tr.fold == k]
    # fp32 master weights: the checkpoint is stored in fp16 and transformers 5 would load it as fp16,
    # which makes AdamW updates underflow -> NaN. Mixed precision comes from autocast below.
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1, dtype=torch.float32).cuda()
    dl = loader(trn.text, (trn.label - mu) / sd, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(dl)
    sched = get_linear_schedule_with_warmup(opt, math.ceil(0.1 * steps), steps)
    for ep in range(a.epochs):
        model.train()
        for b in dl:
            b = {kk: v.cuda() for kk, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            loss = torch.nn.functional.mse_loss(logits.float().squeeze(-1), b["labels"].float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad()
        pv = predict(model, val.text)  # logged only - epochs are fixed, not selected on this fold
        print(f"fold {k} epoch {ep + 1}: val RMSE {np.sqrt(np.mean((pv - val.label.values) ** 2)):.3f}", flush=True)
    oof[(tr.fold == k).values] = pv
    test_pred += predict(model, te.text) / tr.fold.nunique()
    del model, opt
    torch.cuda.empty_cache()

rmse, r = np.sqrt(np.mean((oof - tr.label.values) ** 2)), np.corrcoef(oof, tr.label.values)[0, 1]
print(f"OOF RMSE {rmse:.3f}  r {r:.3f}")
pd.concat([pd.DataFrame({"filename": tr.filename, "split": "train", "deberta_pred": oof}),
           pd.DataFrame({"filename": te.filename, "split": "test", "deberta_pred": test_pred})]).to_csv(a.out, index=False)
print("TEXT_MODEL_DONE", a.out)
