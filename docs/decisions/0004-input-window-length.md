# 0004 — Input window length: deferred until IndicVoices is profiled

**Date:** 2026-09-09
**Status:** open — do not build the mel cache until this closes

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

## What closes this

Profile `ai4bharat/IndicVoices` (spontaneous/extempore, so expected to be longer) with
`kaggle/01a_corpus_distribution_cpu.py`, then compute the padding cost of each option
against the **source mix weighted** distribution. Free, CPU, ~5 minutes.

Until then the cache is not written, because rewriting 11.5 GB of shards is a GPU-hour
expense and this is a free CPU-hour question.
