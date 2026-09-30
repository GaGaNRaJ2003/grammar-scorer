"""Audio-only representation: time-averaged hidden states of a frozen pretrained speech encoder.

--kind wavlm   : WavLM (base-plus / large), whole clip in one pass.
--kind whisper : Whisper encoder; it only takes 30 s windows, so clips are split into 30 s chunks and
                 only the frames covering real audio (50 frames/s) are averaged - padding is ignored.
Every layer is saved, so the layer choice is made later on CPU in CV (see src/audio_model.py).
Output: .npz with names (split/filename) and emb [n_clips, n_layers, dim].
"""
import argparse, sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/Dataset_Final")
p.add_argument("--kind", choices=["wavlm", "whisper"], default="wavlm")
p.add_argument("--model", default="microsoft/wavlm-base-plus")
p.add_argument("--out", default="outputs/wavlm_emb.npz")
a = p.parse_args()

free, total = torch.cuda.mem_get_info()  # Guard: never share a GPU slice with someone else's job.
if total - free > 2 * 1024**3:
    sys.exit(f"GPU slice already in use ({(total - free) / 1024**3:.1f} GB used) - exiting")

fe = AutoFeatureExtractor.from_pretrained(a.model)
if a.kind == "wavlm":
    from transformers import WavLMModel
    model = WavLMModel.from_pretrained(a.model, dtype=torch.float32).cuda().eval()  # checkpoints are fp16
else:
    from transformers import WhisperModel
    model = WhisperModel.from_pretrained(a.model, dtype=torch.float16).get_encoder().cuda().eval()


def load(f):
    x, sr = sf.read(str(f), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return x if sr == 16000 else librosa.resample(x, orig_sr=sr, target_sr=16000)


@torch.no_grad()
def embed(x):
    if a.kind == "wavlm":
        hs = model(fe(x, sampling_rate=16000, return_tensors="pt").input_values.cuda(),
                   output_hidden_states=True).hidden_states
        return torch.stack([h[0].float().mean(0) for h in hs])
    sums, n = None, 0
    for s in range(0, len(x), 30 * 16000):
        chunk = x[s:s + 30 * 16000]
        feats = fe(chunk, sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
        hs = model(feats, output_hidden_states=True).hidden_states
        k = max(1, int(np.ceil(len(chunk) / 16000 * 50)))  # encoder frames that cover real audio
        part = torch.stack([h[0, :k].float().sum(0) for h in hs])
        sums, n = (part if sums is None else sums + part), n + k
    return sums / n


files = [(s, f) for s in ("train", "test") for f in sorted((Path(a.data) / s).glob("*.wav"))]
names, embs = [], []
for i, (s, f) in enumerate(files, 1):
    embs.append(embed(load(f)).cpu().numpy())
    names.append(f"{s}/{f.name}")
    if i % 100 == 0:
        print(f"{i}/{len(files)}", flush=True)
np.savez_compressed(a.out, names=np.array(names), emb=np.stack(embs).astype(np.float32))
print("EMBED_DONE", a.out, np.stack(embs).shape)
