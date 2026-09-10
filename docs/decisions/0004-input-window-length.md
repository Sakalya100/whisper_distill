# 0004 — Input window length: 10 s provisionally, pending the scraped corpus

**Date:** 2026-09-09 · updated 2026-09-10
**Status:** open — **provisionally 10 s**; do not build the mel cache until this closes

## Why this is a decision and not a parameter

Whisper pays the **full window cost on every step regardless of how much of it is speech**.
A 2.6 s clip in a 10 s window costs exactly what a 10 s clip costs. So window length sets
the price of every training step and every on-device inference, and it is baked into the
cache the moment the mel shards are written.

## What the data says so far

Vaani Hindi, 400 rows measured:

| | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|
| seconds | 1.71 | 2.13 | **2.60** | 3.79 | 6.35 | 14.61 |

2.0% exceed 10 s. A 10 s window is ~74% padding at the median.

## The tension

Sizing the window to the **training corpus** and sizing it to the **target task** pull in
opposite directions:

- Vaani's median is 2.6 s. A 4 s window would cover ~p78 of it and cost 40% of a 10 s
  window's encoder compute.
- Real dictation — "remind me to call mom at 4 tomorrow, and add milk to the list" — runs
  3–10 s. A 4 s window would truncate the actual use case, and the model would never have
  seen a long utterance in training.

**The window must serve the task, not the corpus.** A model that is cheap to train and
fails on real input is not a saving. So the resolution is not to shrink toward Vaani.

## Options, to be decided on the combined distribution

1. **Fixed 10 s.** Matches the task, wastes ~74% of the compute on Vaani clips. Simplest,
   and window length is already one of the three planned ablations.
2. **Length-bucketed training.** Bucket the cache by duration and train with two window
   sizes — say 4 s and 10 s — batching within a bucket. The encoder's positional embeddings
   are sinusoidal, so any prefix is valid and slicing per bucket is architecturally cheap.
   Recovers most of the wasted compute and arguably improves robustness by exposing the
   model to both lengths. Costs complexity, and stacks a second off-distribution change on
   top of the 30 s → short-window change Whisper never saw.
3. **Fixed 6 s.** Covers ~p88 of Vaani and most short dictation, 60% of the 10 s compute.
   Truncates long dictation, which is the failure users would notice.

## Measured 2026-09-10 — IndicVoices is profiled

Both referenced sources are now measured. See the 2026-09-10 addendum in
[the corpus probe note](../research/2026-09-09-corpus-schema-probe.md); raw profiles live
in `docs/research/profiles/`.

```
python -m whisper_distill.data.window docs/research/profiles/corpus_profile_*.json
```

As segmented — what `01_acquire_segment_cpu.py` actually builds, since `merge_to_window`
splits an over-long region into back-to-back windows rather than truncating it:

| window | padding | speech dropped | windows/clip | corpus-pass compute |
|---|---|---|---|---|
| 4 s | 26.3% | 1.60% | 1.33 | 0.52x |
| 6 s | 42.4% | 0.96% | 1.14 | 0.67x |
| 8 s | 53.6% | 0.73% | 1.07 | 0.83x |
| **10 s** | 61.3% | 0.54% | 1.03 | 1.00x |

**Data loss does not discriminate.** Every candidate discards under 2% of speech, because
long clips are split rather than cut. The tension stated above was framed on the
assumption that a short window truncates the corpus; for *training* it does not. So the
question reduces to compute against task fit — and the task-fit argument above is
unchanged and still decisive. Dictation runs 3–10 s. A 6 s window would truncate real user
input at inference, where there is no `merge_to_window` to split it, and the 33% compute
saving does not buy back a model that cuts users off mid-sentence.

**Option 1 (fixed 10 s) is therefore the working choice.** Option 2 (length bucketing)
stays open as an efficiency ablation, and it is more attractive than it looked: at 1.03
windows per clip the corpus is overwhelmingly single-window, so bucketing is mostly a
question of how to batch the short tail. Option 3 (fixed 6 s) is rejected — it optimises
the corpus at the task's expense.

## What still closes this

Two things, and neither is expensive:

1. **The scraped Hinglish corpus, 30 h of the 200 h plan, is unmeasured** and is the source
   most likely to be long-form. It is also — per the 2026-09-10 addendum — now load-bearing
   for the Hinglish behaviour itself, since IndicVoices Hindi turned out to be 4.25%
   Latin-script and cannot supply it. Profile it with `01a` once it exists.
2. **The post-VAD distribution**, which is what actually reaches the cache. The table above
   applies `merge_to_window`'s splitting to raw clip durations; the real pipeline runs VAD
   first, so leading and trailing silence is trimmed before splitting. That moves padding
   down and nothing else. `01_acquire_segment_cpu.py` already prints mean clip length —
   read it from the first real acquisition run.

Until then the cache is not written, because rewriting 11.5 GB of shards is a GPU-hour
expense and this is a free CPU-hour question.
