# Grammar Scoring Engine for Spoken English

Solution for the **SHL Hiring Assessment 2026** Kaggle competition: predict a 1–5 grammar score
(averaged rater MOS) for 45–60 s spoken English answers.

**Main deliverable:** [`grammar_scoring_engine.ipynb`](grammar_scoring_engine.ipynb). It contains the approach,
data findings, ablation table, training/CV metrics, error analysis, and a check that it reproduces the submitted file.

## Approach

```
 wav ─► faster-whisper large-v3 ─► transcript + word timings
  │                                 ├─► fluency + text features, CoLA acceptability, grammar-correction edit rate
  │                                 ├─► TF-IDF ─► ridge
  │                                 └─► DeBERTa-v3-large regressor (fine-tuned, 5 folds × 3 seeds)
  └─► Whisper-large-v3 encoder + LoRA adapters (fine-tuned, 5 folds) ─► regression head
          out-of-fold predictions ─► non-negative weighted average (level 2, nested CV)
```

The final model combines **what was said** (grammar-aware text models on the transcript) with **how it was said**
(the Whisper speech encoder, adapted to the task with LoRA). Every level-1 model predicts out-of-fold on fixed, batch-stratified folds. The level-2
weights are evaluated with nested CV.

## Results

| Model | CV RMSE | CV Pearson r | Test-mix CV RMSE | Public LB (RMSE) |
|---|---|---|---|---|
| Mean predictor | 1.014 | – | – | – |
| **Final v9 (4-part stack, LoRA-tuned Whisper encoder)** | **0.516** | **0.861** | **0.510** | **0.3392** |
| Same with frozen Whisper encoder + ridge | 0.518 | 0.860 | 0.515 | 0.3443 |
| v8 (7-part stack, frozen encoders) | 0.515 | 0.862 | 0.518 | 0.3401 |

Test-mix CV RMSE re-weights CV errors to the test set's mix of recording batches.

Key data findings (details in the notebook):
1. **37 zero labels are missing ratings.** They are the appended block `audio_5037…5073` (normal files are ids 0–784), a score outside the 1–5
   rubric, on ordinary speech. They are excluded from training.
2. **Two recording batches.** ~57 % of test clips come from the ~45 s batch, versus ~17 % of training clips.
   Features are length-normalised, and validation and the final fit are importance-weighted to the test mix.
3. **Whisper removes filled pauses** ("um", "uh"), so the audio branch carries hesitation information that the
   transcript loses.

## Repository layout

| Path | Purpose |
|---|---|
| `grammar_scoring_engine.ipynb` | Report + CPU pipeline (audio regressors, stacking, evaluation, submission) |
| `src/transcribe_fw.py` | ASR: faster-whisper large-v3 with word timestamps and confidences |
| `src/features.py` | Length-normalised fluency / text features |
| `src/grammar_features.py` | CoLA acceptability + CoEdIT grammar-correction edit rate |
| `src/text_model.py` | DeBERTa-v3 fine-tuning: out-of-fold + test predictions |
| `src/audio_finetune.py` | LoRA fine-tuning of the Whisper-large-v3 encoder: out-of-fold + test predictions |
| `src/audio_embed.py` | Frozen speech-encoder embeddings (Whisper encoder / WavLM), used in the ablation |
| `src/audio_model.py` | Ridge on embeddings: out-of-fold + test predictions |
| `src/train.py` | CV, stacking (`--stack`), batch weighting (`--weighted`), submission |
| `jobs/*.pbs` | PBS GPU jobs used on our cluster (see note below) |
| `jobs/setup_env.sh`, `jobs/setup_fw.sh` | Environment setup |
| `jobs/dl_model.py`, `jobs/to_safetensors.py` | Model download / `.bin` → safetensors conversion (transformers 5 + torch < 2.6) |
| `jobs/merge_asr.py` | Merges the two parallel ASR job outputs |
| `jobs/build_notebook.py` | Generates the notebook |
| `results/` | Final submission file (`sub06_…csv`: filename + predicted score) and `submissions.csv`, the log of every Kaggle submission with its CV and public-leaderboard score |
| `data/`, `outputs/` | Competition data and generated artefacts. **Not committed**: the competition rules forbid redistributing the data or anything derived from it. |

## Reproduce

1. **Environment:** Python 3.11 with `requirements.txt`. torch/torchaudio come from
   `https://download.pytorch.org/whl/cu121`. See `jobs/setup_env.sh` and `jobs/setup_fw.sh`.
2. **Data:** download the competition data into `data/Dataset_Final/` (`train/`, `test/`, `train.csv`, `test.csv`).
3. **GPU steps**, in order (any ≥10 GB GPU works; DeBERTa-large is faster on ≥24 GB):
   ```bash
   python src/transcribe_fw.py                     # -> outputs/transcripts_fw_large_v3.jsonl
   python src/grammar_features.py --transcripts outputs/transcripts_fw_large_v3.jsonl
   python src/train.py --labels data/Dataset_Final/train.csv   # first run also writes outputs/folds.csv
   for s in 0 100 200; do python src/text_model.py --model microsoft/deberta-v3-large --lr 1e-5 --seed $s \
       --transcripts outputs/transcripts_fw_large_v3.jsonl --out outputs/deberta_large_s$s.csv; done
   python src/audio_finetune.py                    # -> outputs/whisper_ft_preds.csv (~2 h on a 40 GB A100 slice)
   python src/audio_embed.py --kind whisper --model openai/whisper-large-v3 --out outputs/whisper_enc_emb.npz   # ablation only
   ```
   (Our runs named the seed-0 file `deberta_large_preds.csv`; the notebook uses those names.)
4. **CPU steps + report:** run `grammar_scoring_engine.ipynb` top to bottom. It writes `outputs/submission_final.csv`
   and checks that it is identical to the submitted final file.

**Cluster note:** the `jobs/*.pbs` files are the PBS batch scripts we used on a shared GPU cluster. Submit them from
the repo root with `qsub -q <your-gpu-queue> -v GPU=<gpu-id> jobs/<name>.pbs`. Each job refuses to start if its GPU is
already in use. Without PBS, run the same `python src/...` commands directly.
