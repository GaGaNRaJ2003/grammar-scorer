"""Hand-crafted fluency / text features from the ASR output (one row per clip).

All counts are normalised (per minute of audio or per 100 words) because train clips are mostly
~60 s while test clips are mostly ~45 s; raw counts would encode clip length, not grammar.
"""
import json
import re
import sys

import numpy as np
import pandas as pd

FILLERS = {"um", "uh", "er", "erm", "ah", "hmm", "mm", "uhm"}
TOKEN = re.compile(r"[a-z']+")


def norm(w):
    return "".join(TOKEN.findall(w.lower()))


def mattr(tokens, window=50):
    """Moving-average type-token ratio: vocabulary variety that doesn't shrink with text length."""
    if len(tokens) <= window:
        return len(set(tokens)) / max(len(tokens), 1)
    return float(np.mean([len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1)]))


def clip_features(r):
    words = [w for w in r["words"] if norm(w[0])]
    toks = [norm(w[0]) for w in words]
    n, dur = len(toks), r["duration"]
    per100 = 100 / max(n, 1)
    starts, ends = np.array([w[1] for w in words]), np.array([w[2] for w in words])
    gaps = starts[1:] - ends[:-1] if n > 1 else np.array([])
    speech = (ends[-1] - starts[0]) if n else 0.0
    sents = [s for s in re.split(r"[.!?]+", r["text"]) if s.strip()]
    slen = np.array([len(TOKEN.findall(s.lower())) for s in sents]) if sents else np.array([0])
    segs = np.array(r["segments"]) if r["segments"] else np.zeros((1, 4))
    probs = np.array([w[3] for w in words]) if n else np.array([0.0])
    return {
        "filename": r["filename"], "split": r["split"], "duration": dur,
        # delivery / fluency
        "n_words": n,
        "words_per_min": n / dur * 60,
        "articulation_rate": n / speech * 60 if speech > 0 else 0.0,  # speed while actually talking
        "speech_ratio": speech / dur,
        "lead_silence": starts[0] if n else dur,
        "pauses_per_min_05": (gaps > 0.5).sum() / dur * 60,
        "pauses_per_min_1": (gaps > 1.0).sum() / dur * 60,
        "mean_pause": gaps[gaps > 0.5].mean() if (gaps > 0.5).any() else 0.0,
        "fillers_per100": sum(t in FILLERS for t in toks) * per100,
        "repeats_per100": sum(a == b for a, b in zip(toks, toks[1:])) * per100,  # "I I think"
        "bigram_repeats_per100": sum((toks[i], toks[i + 1]) == (toks[i + 2], toks[i + 3])
                                     for i in range(max(n - 3, 0))) * per100,  # "I was I was"
        # ASR confidence: low confidence often marks mispronounced / ill-formed stretches
        "word_prob_mean": probs.mean(),
        "word_prob_low_frac": (probs < 0.5).mean(),
        "seg_logprob_mean": segs[:, 2].mean(),
        "no_speech_prob_max": segs[:, 3].max(),
        # text shape
        "n_sentences_per_min": len(sents) / dur * 60,
        "sent_len_mean": slen.mean(),
        "sent_len_std": slen.std(),
        "sent_len_max": slen.max(),
        "mattr": mattr(toks),
        "word_len_mean": np.mean([len(t) for t in toks]) if n else 0.0,
        "long_word_frac": np.mean([len(t) >= 7 for t in toks]) if n else 0.0,
    }


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return pd.DataFrame([clip_features(r) for r in rows]), pd.DataFrame(rows)[["filename", "split", "text"]]


if __name__ == "__main__":
    f, _ = load(sys.argv[1] if len(sys.argv) > 1 else "outputs/transcripts_fw_large_v3.jsonl")
    assert f.filter(like="per_min").ge(0).all().all() and f.speech_ratio.between(0, 1.01).all()
    print(f.describe().T.round(2).to_string())
