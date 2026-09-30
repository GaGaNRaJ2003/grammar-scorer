"""Grammar-specific features per clip from two pretrained models (GPU, frozen, no training):

1. CoLA acceptability (RoBERTa-large trained on the Corpus of Linguistic Acceptability):
   P(sentence is grammatical) for every sentence-like chunk -> mean / min / low quantile / share < 0.5.
2. Grammatical error correction (CoEdIT-large): minimally correct each chunk, then count word edits
   -> edits per 100 words, split into insertions / deletions / replacements.

Whisper barely punctuates spontaneous speech, so long "sentences" are cut into <= 25-word chunks.
Outputs outputs/grammar_features.csv and outputs/gec_corrections.jsonl (for error analysis).
"""
import argparse, difflib, json, re, sys

import numpy as np
from pathlib import Path

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--transcripts", default="outputs/transcripts_fw_large_v3.merged.jsonl")
p.add_argument("--out", default="outputs/grammar_features.csv")
p.add_argument("--limit", type=int, default=0)
p.add_argument("--raw", default="outputs/grammar_raw.json",
               help="cache of model outputs; if it exists the GPU models are skipped (features recomputed on CPU)")
a = p.parse_args()

MAX_WORDS = 25


def chunks(text):
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        w = s.split()
        out += [" ".join(w[i:i + MAX_WORDS]) for i in range(0, len(w), MAX_WORDS)]
    return [c for c in out if len(c.split()) >= 3]


def words(text):
    """Lower-case words without punctuation: the corrector also adds punctuation/capitals, which
    Whisper omits in spontaneous speech - those must not count as grammar edits."""
    return re.findall(r"[a-z0-9']+", text.lower())


def edits(src, tgt):
    s, t = words(src), words(tgt)
    ops = {"insert": 0, "delete": 0, "replace": 0}
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, s, t, autojunk=False).get_opcodes():
        if op != "equal":
            ops[op] += max(i2 - i1, j2 - j1)
    return ops


rows = [json.loads(l) for l in open(a.transcripts, encoding="utf-8")][:a.limit or None]
clip_chunks = [chunks(r["text"]) for r in rows]
flat = [c for cs in clip_chunks for c in cs]
print(f"{len(rows)} clips, {len(flat)} chunks", flush=True)

raw = Path(a.raw)
if raw.exists():
    cache = json.loads(raw.read_text(encoding="utf-8"))
    assert cache["flat"] == flat, "grammar_raw.json was built from different transcripts - delete it"
    acc, fixed = cache["acc"], cache["fixed"]
else:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    free, total = torch.cuda.mem_get_info()  # Guard: never share a GPU slice with someone else's job.
    if total - free > 2 * 1024**3:
        sys.exit(f"GPU slice already in use ({(total - free) / 1024**3:.1f} GB used) - exiting")

    cola = pipeline("text-classification", model="cointegrated/roberta-large-cola-krishna2020",
                    device=0, dtype=torch.float16, top_k=None)
    acc = [next(d["score"] for d in r if d["label"] == "LABEL_0")  # LABEL_0 = acceptable
           for r in cola(flat, batch_size=64, truncation=True)]
    print("cola done", flush=True)

    tok = AutoTokenizer.from_pretrained("grammarly/coedit-large")
    gec = AutoModelForSeq2SeqLM.from_pretrained("grammarly/coedit-large", dtype=torch.float16).cuda().eval()
    fixed = []
    for i in range(0, len(flat), 64):
        batch = [f"Fix grammatical errors in this sentence: {c}" for c in flat[i:i + 64]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=96).to("cuda")
        with torch.no_grad():
            out = gec.generate(**enc, max_new_tokens=96, num_beams=4)
        fixed += tok.batch_decode(out, skip_special_tokens=True)
        print(f"gec {min(i + 64, len(flat))}/{len(flat)}", flush=True)
    # save model outputs before feature maths, so a later bug never costs GPU time again
    raw.write_text(json.dumps({"flat": flat, "acc": acc, "fixed": fixed}), encoding="utf-8")

feats, k = [], 0
with open("outputs/gec_corrections.jsonl", "w", encoding="utf-8") as fh:
    for r, cs in zip(rows, clip_chunks):
        n = len(cs)
        pa, src, tgt = np.array(acc[k:k + n] or [np.nan]), cs, fixed[k:k + n]
        k += n
        ops = [edits(s, t) for s, t in zip(src, tgt)]
        n_words = sum(len(words(s)) for s in src) or 1
        tot = {o: sum(d[o] for d in ops) for o in ("insert", "delete", "replace")}
        feats.append({
            "filename": r["filename"], "split": r["split"], "n_chunks": n,
            "cola_mean": np.nanmean(pa), "cola_min": np.nanmin(pa), "cola_q10": np.nanquantile(pa, 0.1),
            "cola_low_frac": np.nanmean(pa < 0.5),
            "gec_edits_per100": sum(tot.values()) / n_words * 100,
            **{f"gec_{o}_per100": v / n_words * 100 for o, v in tot.items()},
            "gec_changed_chunk_frac": np.mean([words(s) != words(t) for s, t in zip(src, tgt)]) if n else np.nan,
        })
        fh.write(json.dumps({"filename": r["filename"], "split": r["split"], "pairs": list(zip(src, tgt))}) + "\n")

pd.DataFrame(feats).to_csv(a.out, index=False)
print("GRAMMAR_DONE", a.out)
