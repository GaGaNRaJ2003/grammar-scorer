"""Transcribe every clip with faster-whisper (CTranslate2) and keep word-level timestamps.

Output: one JSON line per clip ->
  {filename, split, duration, text, words: [[word, start, end, prob], ...],
   segments: [[start, end, avg_logprob, no_speech_prob], ...]}
Word timings -> pause / speech-rate features; confidences -> ASR-reliability and no-speech signals.
"""
import argparse, json, sys, time
from pathlib import Path

import librosa
import soundfile as sf
from faster_whisper import WhisperModel

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/Dataset_Final")
p.add_argument("--out", default="outputs/transcripts_fw_large_v3.jsonl")
p.add_argument("--model", default="Systran/faster-whisper-large-v3")
p.add_argument("--limit", type=int, default=0, help="only process N clips (smoke test / timing)")
p.add_argument("--reverse", action="store_true", help="work from the end (2nd job on another GPU, merge after)")
a = p.parse_args()


def load(f):
    """Mono float32 at 16 kHz (no ffmpeg on the cluster, so decode with soundfile)."""
    x, sr = sf.read(str(f), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return x if sr == 16000 else librosa.resample(x, orig_sr=sr, target_sr=16000)


import torch  # only for the guard: never share a GPU slice with someone else's job
free, total = torch.cuda.mem_get_info()
if total - free > 2 * 1024**3:
    sys.exit(f"GPU slice already in use ({(total - free) / 1024**3:.1f} GB used) - exiting")

model = WhisperModel(a.model, device="cuda", compute_type="float16")

files = [(s, f) for s in ("train", "test") for f in sorted((Path(a.data) / s).glob("*.wav"))]
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
done = {(r["split"], r["filename"]) for r in map(json.loads, out.open())} if out.exists() else set()
todo = [(s, f) for s, f in (files[::-1] if a.reverse else files) if (s, f.name) not in done][:a.limit or None]
print(f"{len(files)} clips, {len(todo)} to do", flush=True)

t0 = time.time()
with out.open("a", encoding="utf-8") as fh:
    for i, (s, f) in enumerate(todo, 1):
        x = load(f)
        # condition_on_previous_text=False avoids hallucination loops; no VAD so pauses stay in the timings
        segs, _ = model.transcribe(x, language="en", beam_size=5, word_timestamps=True,
                                   condition_on_previous_text=False)
        segs = list(segs)
        fh.write(json.dumps({
            "filename": f.name, "split": s, "duration": len(x) / 16000,
            "text": " ".join(g.text.strip() for g in segs),
            "words": [[w.word, w.start, w.end, round(w.probability, 4)] for g in segs for w in g.words],
            "segments": [[g.start, g.end, round(g.avg_logprob, 4), round(g.no_speech_prob, 4)] for g in segs],
        }) + "\n")
        if i % 10 == 0 or i == len(todo):
            fh.flush()
            print(f"{i}/{len(todo)}  {(time.time() - t0) / i:.2f}s/clip", flush=True)
print("TRANSCRIBE_DONE")
