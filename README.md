# whisper_distill

Distil a ~135 MB on-device Hinglish **dictation** model from a Hindi Whisper-medium
teacher, trained entirely inside Kaggle's free 30-hour weekly GPU quota.

Not an ASR benchmark chase. The target is the thing a dictation user actually judges:
punctuation, casing, and whether "chaar baje" comes out as "4 baje".

**Plan:** [`docs/01-kaggle-execution-plan.md`](docs/01-kaggle-execution-plan.md) ·
[published page](https://claude.ai/code/artifact/69fb3c75-b71a-4917-b987-2c6b851b32c1)

---

## Where it stands

The plan is ordered by what has to be measured before the rest of it means anything.

| Gate | Question | Status |
|---|---|---|
| 0 | Does a T4×2 session bill wall-clock or GPU-hours? | **measuring** — probe running |
| 1 | Does the teacher preserve Latin-script English on code-mixed speech? | not started |

Gate 0 moves the schedule ~2×. Gate 1 can invalidate the teacher, and with it the
80-mel single-pass feature decision. Neither is expensive; both are blocking.

Confirmed so far: phone verified · 30 h GPU + 20 h TPU pools · 200 GiB private dataset
quota · **≥2 concurrent T4×2 sessions allowed**.

## Target model

| Change | Params after |
|---|---:|
| whisper-small Hindi baseline | ~244 M |
| Decoder 12 → 4 layers (maximally spaced copy) | ~160 M |
| Vocabulary pruned ~51.8k → ~16k | ~132 M |
| 10 s window (positional embeddings sliced) | ~132 M, ~3× less encoder compute |

INT8 → ~135 MB, which runs on a 4 GB phone. Encoder stays whole and **frozen**: no
gradients or optimiser state for ~88 M params roughly doubles usable batch size on a T4.

## Layout

```
docs/            research write-ups, decision records, published materials
  decisions/     one file per choice that would be expensive to revisit
  research/      dated findings with sources, append-only
kaggle/          paste-ready notebook scripts, one per pipeline stage
src/whisper_distill/
  config.py      every number that appears in the plan, in one place
  data/          acquisition, VAD segmentation, shard packing      (CPU, free)
  labeling/      teacher wrappers, quota decoder, label filters    (GPU)
  modeling/      student surgery: decoder cut, vocab prune, window slice
  training/      distillation loss and the vocab-pruning guard
  evaluation/    WER/CER, entity accuracy, false-trigger rate      (CPU, free)
  itn/           rule-based inverse text normalisation             (CPU, no model)
eval_set/        spec + manifests for the self-recorded dictation eval set
tests/
```

## Pipeline

```
CPU  (free)   stream Vaani/IndicVoices → resample → silero VAD → ≤10 s clips
GPU  (quota)  teacher audit → fused pass: pseudo-label + mel cache in one read
CPU  (free)   WER filter (referenced) / entropy filter (scraped) → shard + index
GPU  (quota)  cut student → train on cached mels → checkpoint to private HF Hub repo
CPU  (free)   ITN post-process → WER/CER/entity/false-trigger report
```

Everything that does not need a GPU runs in a CPU session, which does not draw on the
30-hour pool. That single split is what makes the budget fit.

## Running it

```bash
pip install -e ".[dev]"        # local: tests, linting
pytest                         # 93 tests, no torch required
```

Inside a Kaggle notebook, see [`kaggle/README.md`](kaggle/README.md) — it carries the
six rules that apply to every session (commit rather than run interactively, emit a
heartbeat, secrets in Add-ons, accept dataset terms first, print `df -h`, never TPU).

## What is verified and what is not

Tests cover the pure logic: layer selection, vocabulary selection, ITN, WER/CER pooling,
entity scoring, shard round-trip, segmentation, label filters, quota decoding.

**Not yet executed:** anything importing torch or transformers —
`modeling/surgery.py`, `modeling/student.py`, `training/distill_loss.py`, and the GPU
notebook scripts. There is no torch in the local dev environment; their first run is the
step-3 smoke test, which exists precisely to catch this for one quota-hour.

## Data and licensing

`ARTPARK-IISc/Vaani-transcription-part` (Hindi, 963 h available) and
`ai4bharat/IndicVoices` are both CC-BY-4.0 and gated — accept the terms on Hugging Face
before running anything. The scraped Hinglish portion is not redistributable; only
manifests and derived features belong in a private dataset, never in this repo.

No audio, features, checkpoints or credentials are committed. See `.gitignore`.
