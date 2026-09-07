# Research pass — free-tier Kaggle as the compute substrate

**Date:** 2026-09-08
**Question:** can this project be executed end-to-end on Kaggle's free tier, and what does
the platform force us to change?

Append-only. Later passes get their own file.

---

## Platform facts

| Fact | Value | Confidence |
|---|---|---|
| GPU quota | 30 h/week, **one pool** shared across P100 and T4×2 | multiple 2026 sources agree |
| TPU quota | 20 h/week, tracked separately | as above |
| Session cap | ~12 h (CPU/GPU), ~9–12 h (TPU) | as above |
| T4×2 | 2× Tesla T4, 16 GB each, 32 GB host RAM | as above |
| P100 | 1× Tesla P100 16 GB, 32 GB host RAM | as above |
| CPU session | 2× Intel Xeon, 32 GB RAM | Kaggle env notes |
| `/kaggle/working` | writeable, persists between runs, **~20 GB cap** | Kaggle env notes |
| `/kaggle/input` | read-only mount for attached datasets | ditto |
| `/kaggle/temp` | scratch, may not persist | ditto |
| Private dataset quota | **200 GiB** -- the UI shows `214.75 GB`, which is the same number in decimal | **confirmed on account** |
| Dataset top-level files | **max 50** — use archives/shards below that | Kaggle docs discussion |
| Background execution | `Save & Run All (Commit)` runs headless | Kaggle docs |
| Phone verification | gates **both** accelerators and notebook internet | **confirmed: Verified** |
| Concurrent GPU sessions | **>=2** -- two T4x2 commits ran side by side | **confirmed on account** |

### Open questions — resolved by `kaggle/00_quota_probe.py`

- `[UNVERIFIED]` **Does a T4×2 session bill wall-clock or GPU-hours?** Not documented
  anywhere I could find, including the floating-quota announcement. ~2× schedule impact.
- `[UNVERIFIED]` **Do CPU-only sessions consume the GPU pool?** Widely assumed not to.
  The entire "push prep to CPU" strategy rests on it, so it gets measured, not assumed.
- `[UNVERIFIED]` **Actual scratch disk on a GPU session.** The 20 GB figure is the *output*
  cap. Print `df -h`.

### Decoding a two-session probe

Because the probe ran as two concurrent T4x2 commits rather than one, the meter delta has
two candidate readings instead of one, for `N` sessions of `T` minutes each:

```
wall-clock billing  ->  delta ~= N * T          (2 x 15 =  30 min)
per-GPU billing     ->  delta ~= N * T * 2      (2 x 15 x 2 = 60 min)
```

Both are still distinguishable, **provided both sessions ran for the same duration** --
read each notebook's reported runtime from its version history rather than assuming the
sleep duration, since a committed run also pays container startup and teardown. A reading
near the 45-minute midpoint is inconclusive and should be re-run as a single 30-minute
commit with nothing else active. `labeling/quota.py` implements this decode and refuses to
guess at the midpoint.

## Platform gotchas found in the wild

- **Kaggle TPU is unusable for this stack.** Importing `transformers` Trainer or pipeline
  classes crashes TPU sessions (HF issue #28609). Reported by two independent Whisper
  fine-tuning repos targeting Kaggle. Do not budget the 20 TPU hours.
- **Trainer's default multi-GPU on T4×2 is naive model parallelism**, which leaves one GPU
  idle for most of each step. Use DDP (`torchrun` / `accelerate`).
- **Beam search spikes VRAM** and OOMs on T4 where greedy fits. Greedy for all labelling.
- **Interactive sessions idle-timeout** well short of 12 h. Commit every long run.
- **Silent cells can be killed** on iopub/per-cell timeouts even while the GPU is busy.
  Emit a heartbeat.
- **File persistence must be enabled before launch** if you intend to resume from
  `/kaggle/working` — but see below, Hub checkpointing is better.

## The checkpoint-survival pattern

Kaggle sessions die. The pattern that actually survives it, per HF forum discussion and the
Kaggle/HF integration:

```python
TrainingArguments(
    push_to_hub=True,
    hub_strategy="all_checkpoints",   # pushes optimiser state too, not just weights
    hub_private_repo=True,
    save_total_limit=1,               # /kaggle/working is capped
    save_steps=500,
)
```

On session start, resume from the Hub rather than from local disk, which does not survive.
`HF_TOKEN` goes in **Add-ons → Secrets** and is read with `UserSecretsClient`.

## Model availability findings

**IndicWhisper is not an HF repo.** Vistaar links Hindi weights as `hindi_models.zip` on
e2enetworks object storage. [Issue #4](https://github.com/AI4Bharat/vistaar/issues/4),
open since April 2024, is someone asking how to load the model in Python at all. Vistaar
itself is MIT-licensed. → [decision 0001](../decisions/0001-teacher-checkpoint.md).

**`vasista22/whisper-hindi-medium`** is a plain HF checkpoint: whisper-medium base, Speech
Lab IIT Madras, Bhashini-funded, trained on GramVaani + ULCA + Shrutilipi + FLEURS.
FLEURS 6.82 / CV11 11.38. `whisper-hindi-small` is the same lineage at small scale
(FLEURS 9.02 / CV11 14.12).

**Risk carried by both:** all four training corpora are Devanagari, normalised and
unpunctuated. This is the Gate 1 audit, and it is project-blocker class rather than a
smoke-test detail — if the teacher transliterates Latin-script English, every pseudo-label
is wrong in the project's core dimension.

**128-mel fallbacks** if Gate 1 fails: `openai/whisper-large-v3`,
`ARTPARK-IISc/whisper-large-v3-vaani-hindi` (fine-tuned on ~718 h transcribed Hindi).
Both forfeit single-pass feature extraction.

## Dataset findings

**Vaani** — the headline "2,043 hours" is the **transcribed subset**, not the corpus. The
full collection is 31,255 h audio + 2,043 h transcriptions + 289,838 images across 165
districts and 105 languages. What we need is `ARTPARK-IISc/Vaani-transcription-part`:

- 59 language configs, 2,041.54 h total, **234 GB total**
- **Hindi: 963.04 h** — the largest config
- CC-BY-4.0, **gated** (accept terms, pass a token — free)
- `load_dataset("ARTPARK-IISc/Vaani-transcription-part", "Hindi", token=...)`

We want ~120 of 963 hours → **stream with early stop**, do not materialise the config.

**IndicVoices** — `ai4bharat/IndicVoices`, CC-BY-4.0, gated, HF-hosted. Built for IndicASR
across all 22 scheduled languages. The derived IndicVoices-R is 93.25% extempore, which
indicates the parent corpus is heavily spontaneous — the right register for dictation.

## Throughput finding — CTranslate2 for the labelling pass

`faster-whisper` (CTranslate2 backend) reports ~4× lower latency and ~half the VRAM versus
the reference implementation; batched inference adds a further 2–4×. Published batched
benchmarks average ~12.5× faster than OpenAI's implementation.

Takes the 20 GPU-hour labelling estimate down to ~6–10.

**Two caveats that keep this honest:**
1. **CT2 exposes no logits.** Fine for bulk cross-entropy labels; unusable for the KL
   ablation, which stays on HF transformers over its 50-hour subset.
2. `int8_float16` is documented as tuned for **Ampere and newer**. T4 is Turing.
   Benchmark `int8_float16` against plain `float16` on 30 minutes before committing.

## Arithmetic corrections

Whisper is **100 mel frames/second** (hop 160 at 16 kHz):

```
30 s → 3000 mel frames → 1500 encoder positions
10 s → 1000 mel frames →  500 encoder positions
```

The spec's "80 × 500" conflated the mel cache with the post-conv encoder length. Correct
cache is 80×1000 → 160 KB/clip fp16 → **11.5 GB** for 72,000 clips, not 5.8 GB.

Useful consequence: a ≤10 s clip's student mel is a **prefix** of the teacher's 30 s padded
mel, so one FFT serves both models — which is what makes the fused labelling pass work.
→ [decision 0002](../decisions/0002-mel-config-and-cache-layout.md).

## Sources

- Kaggle: [floating GPU quota](https://www.kaggle.com/product-feedback/173129) · [private dataset quota doubling](https://www.kaggle.com/product-announcements/512322) · [environment notes compilation](https://huggingface.co/datasets/John6666/knowledge_base_md_for_rag_1/blob/main/kaggle_20251121.md)
- Kaggle + Whisper in practice: [phineas-pta/fine-tune-whisper-vi](https://github.com/phineas-pta/fine-tune-whisper-vi/blob/main/README.md) · [allandclive/fine-tune-whisper-lg](https://github.com/allandclive/fine-tune-whisper-lg) · [vasistalodagala/whisper-finetune](https://github.com/vasistalodagala/whisper-finetune)
- Checkpointing: [HF × Kaggle integration](https://huggingface.co/blog/kaggle-integration)
- Models: [Vistaar](https://github.com/AI4Bharat/vistaar) · [issue #4](https://github.com/AI4Bharat/vistaar/issues/4) · [whisper-hindi-medium](https://huggingface.co/vasista22/whisper-hindi-medium) · [whisper-hindi-small](https://huggingface.co/vasista22/whisper-hindi-small) · [whisper-large-v3-vaani-hindi](https://huggingface.co/ARTPARK-IISc/whisper-large-v3-vaani-hindi)
- Data: [Vaani-transcription-part](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part) · [ai4bharat/IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) · [IISc × HF collaboration](https://github.com/huggingface/blog/blob/main/iisc-huggingface-collab.md)
- Distillation method: [Distil-Whisper training README](https://github.com/huggingface/distil-whisper/blob/main/training/README.md) · [paper](https://arxiv.org/abs/2311.00430)
- Throughput: [batched Whisper](https://mobiusml.github.io/batched_whisper_blog/) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
