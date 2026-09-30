"""Fine-tune the Whisper-large-v3 encoder (LoRA) to regress the grammar score directly from audio.

The frozen encoder + ridge (src/audio_embed.py + src/audio_model.py) was our best single component; here the
encoder adapts to the task through low-rank adapters on its attention projections (<1 % of weights), with a
small head on the time-averaged last layer. Clips are cut into <=30 s windows (Whisper's input size) and pooled
over real-audio frames only. Same saved CV folds and a fixed number of epochs, so the out-of-fold predictions
are honest inputs for the stack in src/train.py.
"""
import argparse, math, sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from transformers import WhisperFeatureExtractor, WhisperModel, get_linear_schedule_with_warmup

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/Dataset_Final")
p.add_argument("--folds", default="outputs/folds.csv")
p.add_argument("--model", default="openai/whisper-large-v3")
p.add_argument("--epochs", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate")
p.add_argument("--head-lr", type=float, default=1e-3)
p.add_argument("--bs", type=int, default=8, help="clips per batch")
p.add_argument("--rank", type=int, default=16)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="outputs/whisper_ft_preds.csv")
a = p.parse_args()

free, total = torch.cuda.mem_get_info()  # Guard: never share a GPU slice with someone else's job.
if total - free > 2 * 1024**3:
    sys.exit(f"GPU slice already in use ({(total - free) / 1024**3:.1f} GB used) - exiting")

# ---- inputs: log-mel windows for every clip (computed once) ----
fe = WhisperFeatureExtractor.from_pretrained(a.model)
lab = pd.read_csv(f"{a.data}/train.csv")
lab = lab[lab.label > 0]  # audio_5037..5073 zeros are missing ratings (see src/train.py)
tr = lab.merge(pd.read_csv(a.folds), on="filename").reset_index(drop=True)
te = pd.read_csv(f"{a.data}/test.csv")[["filename"]]


def windows(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    x = x if sr == 16000 else librosa.resample(x, orig_sr=sr, target_sr=16000)
    chunks = [x[s:s + 30 * 16000] for s in range(0, min(len(x), 60 * 16000), 30 * 16000)]
    mel = fe(chunks, sampling_rate=16000, return_tensors="pt").input_features.half()  # [n_win, 128, 3000]
    frames = [min(1500, max(1, math.ceil(len(c) / 16000 * 50))) for c in chunks]  # encoder frames of real audio
    return mel, frames


X_tr = [windows(f"{a.data}/train/{f}") for f in tr.filename]
X_te = [windows(f"{a.data}/test/{f}") for f in te.filename]
mu, sd = tr.label.mean(), tr.label.std()
print(f"features ready: {len(X_tr)} train / {len(X_te)} test clips", flush=True)


class Scorer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        enc = WhisperModel.from_pretrained(a.model, dtype=torch.float32).get_encoder()
        enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.enc = get_peft_model(enc, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.1,
                                                  target_modules=["q_proj", "v_proj"]))
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(1280), torch.nn.Dropout(0.1), torch.nn.Linear(1280, 1))

    def forward(self, batch):
        mel = torch.cat([m for m, _ in batch]).cuda()
        h = self.enc(input_features=mel.float()).last_hidden_state  # [n_windows, 1500, 1280]
        pooled, i = [], 0
        for m, frames in batch:  # frame-weighted mean over each clip's windows, padding excluded
            s = sum(h[i + j, :k].sum(0) for j, k in enumerate(frames))
            pooled.append(s / sum(frames))
            i += len(frames)
        return self.head(torch.stack(pooled).float()).squeeze(-1)


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = []
    for i in range(0, len(X), a.bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(X[i:i + a.bs]).float().cpu())
    return torch.cat(out).numpy() * sd + mu


oof, test_pred = np.zeros(len(tr)), np.zeros(len(te))
rng = np.random.default_rng(a.seed)
for k in sorted(tr.fold.unique()):
    torch.manual_seed(a.seed + k)
    trn, val = np.where(tr.fold != k)[0], np.where(tr.fold == k)[0]
    y = torch.tensor(((tr.label - mu) / sd).values, dtype=torch.float32)
    model = Scorer().cuda()
    params = [{"params": [q for q in model.enc.parameters() if q.requires_grad], "lr": a.lr},
              {"params": model.head.parameters(), "lr": a.head_lr}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(trn) / a.bs)
    sched = get_linear_schedule_with_warmup(opt, math.ceil(0.1 * steps), steps)
    for ep in range(a.epochs):
        model.train()
        order = rng.permutation(trn)
        for i in range(0, len(order), a.bs):
            idx = order[i:i + a.bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model([X_tr[j] for j in idx])
            loss = torch.nn.functional.mse_loss(pred.float(), y[idx].cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_([q for g in params for q in g["params"]], 1.0)
            opt.step(), sched.step(), opt.zero_grad()
        pv = predict(model, [X_tr[j] for j in val])  # logged only - epochs are fixed, not selected on this fold
        print(f"fold {k} epoch {ep + 1}: val RMSE {np.sqrt(np.mean((pv - tr.label.values[val]) ** 2)):.3f}", flush=True)
    oof[val] = pv
    test_pred += predict(model, X_te) / tr.fold.nunique()
    del model, opt
    torch.cuda.empty_cache()

rmse, r = np.sqrt(np.mean((oof - tr.label.values) ** 2)), np.corrcoef(oof, tr.label.values)[0, 1]
print(f"OOF RMSE {rmse:.3f}  r {r:.3f}")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pd.concat([pd.DataFrame({"filename": tr.filename, "split": "train", "whisper_ft_pred": oof}),
           pd.DataFrame({"filename": te.filename, "split": "test", "whisper_ft_pred": test_pred})]).to_csv(a.out, index=False)
print("AUDIO_FT_DONE", a.out)
