# Research pass — Vaani Hindi schema and first-row probe

**Date:** 2026-09-09
**Question:** does the acquisition script's schema assumption hold, and what does the
corpus actually look like?

Append-only. See [2026-09-08](2026-09-08-kaggle-free-tier-notes.md) for the platform pass.

---

## Confirmed

Ran a one-row streaming probe against `ARTPARK-IISc/Vaani-transcription-part:Hindi:train`
on a free Kaggle CPU session.

```
columns    : ['audio', 'district', 'gender', 'language', 'referenceImage', 'state', 'transcript']
audio field: AudioDecoder
transcript : 'बहुत ही सुन्दर टमाटर है ।'
decoded via audio_decoder: 2.0s
```

- **Columns match the dataset card exactly.** `transcript` is present and non-empty, so
  the WER filter in step 2 has something to filter against.
- **Gating works** on a read-scoped token accepted through the dataset page.
- **Kaggle's image runs `datasets` 4.x and torchcodec works.** The audio field arrives as
  a torchcodec `AudioDecoder` and `decode_audio_field` handles it natively.

### Decision: use the AudioDecoder path, drop the `decode=False` cast

The script originally cast to `Audio(decode=False)` to bypass torchcodec, on the reasoning
that torchcodec needs a matching torch build plus system FFmpeg and Kaggle's version is
not ours to control. The probe makes that moot **and** makes the cast the riskier option:
`decode=False` routes through soundfile, which nothing has exercised against Vaani's actual
audio format. Prefer the branch with evidence behind it. `audio_io.py` still handles all
four shapes, so a future image change degrades rather than breaks.

## Two things the probe raised that matter more than the schema

### 1. The clip is 2.0 seconds, carrying one complete short utterance

`बहुत ही सुन्दर टमाटर है ।` — "it's a very beautiful tomato." That is not a fragment; it is
a whole utterance. **Vaani's transcribed part appears to be pre-segmented into short
utterances.** If that is typical rather than an outlier, two plan assumptions break:

- **A 10 s student window would be ~80% padding.** That wastes encoder compute on every
  step, and worse, the model never sees a long utterance in training — while real dictation
  ("remind me to call mom at 4 tomorrow, and add milk to the list") runs 3–10 s. Tuning the
  window to 2 s clips would build a model that fails on the actual task.
- **VAD segmentation would have almost nothing to do.** A whole CPU stage idling, and
  `merge_to_window`'s `max_gap_s` tuning would be noise rather than signal.

### 2. The transcript is pure Devanagari with no Latin script

If Vaani Hindi is not meaningfully code-mixed, it is good acoustic and speaker coverage but
it is **not** the source of the Hinglish behaviour this project exists for. That would mean
rebalancing the 120/50/30 hour split toward IndicVoices and scraped audio — and, more
immediately, that **Gate 1's code-switch audit cannot be run on Vaani clips**, because
there would be no code-switching in them to test against.

## Status: n=1 decides nothing

Both observations come from a single row. `kaggle/01a_corpus_distribution_cpu.py` profiles
400 rows (~3–5 min, free) and reports duration deciles, transcript length, Latin-script
rate and code-mix density, with an explicit read-out of what each outcome implies for the
window design and the data split.

**Run 01a before 01.** Getting 5 hours of the wrong-shaped audio costs nothing in quota but
sends step 2 down a path chosen on one sample.
