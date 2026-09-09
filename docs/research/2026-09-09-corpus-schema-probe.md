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

## Measured over 400 rows — one conclusion above was wrong

Profiled 400 rows (0.377 h of audio) in 0.3 min on a free CPU session.

### Audio duration

| | p10 | p25 | median | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|---|
| seconds | 1.71 | 2.13 | **2.60** | 3.79 | 6.35 | 14.61 | 3.40 |

Only **2.0%** (8/400) exceed 10 s. All 400 decoded via the `audio_decoder` path.

**The short-utterance observation holds.** Median 2.60 s against a 10 s window is ~74%
padding — and padding is not free, because Whisper pays the full window cost regardless of
how much of it is speech. There is a real tail though: p90 is 6.35 s and the max is 14.6 s,
so the corpus is not uniformly tiny.

### Transcripts

| | value |
|---|---|
| empty | **0 / 400** |
| median words | 8 |
| **contains Latin script** | **273 / 400 (68.2%)** |
| contains danda `।` | 307 / 400 |
| **contains comma** | **0 / 400** |
| mean code-mix density | 0.157 |
| buckets | hindi_dominant 192 · balanced 181 · dense 27 |

### Correction: Vaani Hindi *is* meaningfully code-mixed

The n=1 note above inferred from one pure-Devanagari transcript that Vaani Hindi might be
essentially monolingual, and that Gate 1's audit therefore could not run on it. **That was
wrong.** 68.2% of transcripts contain Latin script, mean code-mix density is 0.157, and 208
of 400 (52%) fall in the balanced-or-dense buckets.

Consequences of the correction:

- The **120 h Vaani allocation stands.** No rebalancing needed on code-mixing grounds.
- **Gate 1's audit can be run on Vaani clips** — filter to the `dense` and `balanced`
  buckets, which `02_audit_and_label_gpu.py` already does via `code_mix_bucket`.
- A cheap source of genuinely code-mixed Hindi audio with references exists, which is
  better than the plan assumed.

### Zero commas out of 400 — this sharpens Gate 1

References carry a sentence-final danda (307/400) and **no commas at all**. So the human
labels have essentially no intra-sentence punctuation, which is precisely the gap
pseudo-labelling is supposed to fill. That makes the Gate 1 question concrete and testable:
**does the teacher emit commas?** If it does, the pseudo-labelling premise is confirmed on
real numbers rather than assumed. If it does not, the premise is weaker than the plan
claims and the writeup must say so.

## Still open: the window length

Do **not** size the window from Vaani. It is pre-segmented short utterances — good acoustic
and speaker coverage, but not utterance-length coverage — while the target task, dictation,
runs 3–10 s. Sizing to a 2.6 s median would build a model that fails on the real input.

Next: profile `ai4bharat/IndicVoices` with the same script (swap `DATASET`/`CONFIG` at the
top) and decide the window from the **combined** distribution weighted by the final source
mix. See [decision 0004](../decisions/0004-input-window-length.md).
