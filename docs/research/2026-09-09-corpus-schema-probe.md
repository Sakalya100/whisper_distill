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

---

## Addendum 2026-09-10 — the soundfile branch is no longer unexercised

Ran the CPU stages locally on macOS (Python 3.13, `datasets` 5.0.1, no torchcodec, no
system FFmpeg) via `scripts/local_dry_run.ipynb`, against the same Vaani Hindi split.

```
decoded via 'soundfile': 2.02 s, mono float32, peak/rms assertions passed
40 rows profiled through the same path
```

**This retires a caveat stated above.** The decision section says the `decode=False` cast
"routes through soundfile, which nothing has exercised against Vaani's actual audio
format." That is now false: both branches of `decode_audio_field` have run against real
Vaani audio — `audio_decoder` on Kaggle (400 rows) and `soundfile` locally (40 rows).

The decision itself is unchanged: keep the `AudioDecoder` path on Kaggle, because it is
what the image hands you and it has the larger sample behind it. What changes is the
fallback's status. `audio_io.py` was written so a future image change would degrade rather
than break, and that claim now rests on evidence instead of on the code being present.

Two smaller things the local run confirmed:

- **VAD really does have nothing to do on Vaani.** silero-vad on a 2.02 s clip returned a
  single speech region spanning `0.00–2.00 s`, which `merge_to_window` passed through
  unchanged. Exactly what a median of 2.60 s predicts, now observed rather than inferred.
- **silero-vad needs torchaudio, not just torch.** Its `hubconf.py` declares
  `dependencies = ['torch', 'torchaudio']` and `utils_vad.py` imports torchaudio at module
  level, so `torch.hub.load` fails without it even though we only call
  `get_speech_timestamps`. Kaggle's image preinstalls it, which is why
  `01_acquire_segment_cpu.py` has never hit this; a fresh local venv does not.

---

## Addendum 2026-09-10 — IndicVoices measured, and it inverts the code-mixing assumption

Profiled `ai4bharat/IndicVoices:hindi:train`, 400 rows (0.627 h), free Kaggle CPU session.
Raw profile: [`profiles/corpus_profile_IndicVoices_hindi.json`](profiles/corpus_profile_IndicVoices_hindi.json).

### The headline is not the window — it is the Latin-script share

| | Vaani Hindi | IndicVoices Hindi |
|---|---:|---:|
| contains Latin script | **68.2%** | **4.25%** |
| mean code-mix density | 0.157 | **0.0025** |
| buckets | hindi_dominant 192 · balanced 181 · dense 27 | hindi_dominant 398 · balanced 2 · **dense 0** |
| contains danda `।` | 307 / 400 | **0 / 400** |
| contains comma | 0 / 400 | 0 / 400 |

**IndicVoices Hindi is essentially monolingual Devanagari.** That is the exact reverse of
the plan's assumption. The correction recorded above concluded that no rebalancing was
needed *because Vaani turned out to be code-mixed*; it never considered that IndicVoices
might be the monolingual one. Consequences:

- **The 50 h IndicVoices allocation no longer buys Hinglish.** It is still worth having,
  but for a different reason than it was budgeted: it is the only source measured so far
  with real utterance-length coverage. Keep it for acoustic, speaker and length diversity.
- **Vaani is now the only referenced code-mixed audio in the plan.** Gate 1's audit has to
  run on Vaani's `dense` + `balanced` buckets; IndicVoices clips cannot serve it at all.
- **The Hinglish behaviour rests on Vaani plus the 30 h of scraped audio**, and the scraped
  portion has not been collected or measured. That is now the riskiest unmeasured thing in
  the data plan, not a nice-to-have.
- **Zero punctuation of any kind.** Vaani at least carries a sentence-final danda 307/400
  times; IndicVoices carries none. Across 800 profiled rows from two corpora there is not
  one comma. Gate 1's question — *does the teacher emit commas?* — is now backed by 800
  rows rather than 400.

The `120 / 50 / 30` hour split should be revisited on these numbers. Not changing
`DataConfig.target_hours` yet: that is a plan decision and it needs the scraped-audio
measurement first.

### Audio duration

| | p10 | p25 | median | p75 | p90 | max | mean | >10 s |
|---|---|---|---|---|---|---|---|---|
| Vaani | 1.71 | 2.13 | 2.60 | 3.79 | 6.35 | 14.61 | 3.40 | 2.0% |
| IndicVoices | 0.64 | 1.50 | **3.52** | 7.82 | **14.11** | **28.92** | 5.64 | **19.75%** |

IndicVoices is the longer corpus, as decision 0004 expected of spontaneous speech, but its
median is still only 3.52 s. The difference is the tail: p90 of 14.1 s against Vaani's
6.35 s, and a fifth of clips over 10 s against Vaani's 2%.

All 400 decoded via `audio_decoder`, consistent with the earlier Kaggle run.

### Window cost on the combined mix

`python -m whisper_distill.data.window docs/research/profiles/corpus_profile_*.json`.
Weighted by contributed training steps: Vaani is 80.6% of steps against IndicVoices' 19.4%,
despite the 120/50 hour split being 70/30 — short clips buy more steps per hour.

As segmented, which is what `01_acquire_segment_cpu.py` actually builds:

| window | padding | speech dropped | windows/clip | corpus-pass compute |
|---|---|---|---|---|
| 4 s | 26.3% | 1.60% | 1.33 | 0.52x |
| 6 s | 42.4% | 0.96% | 1.14 | 0.67x |
| 8 s | 53.6% | 0.73% | 1.07 | 0.83x |
| **10 s** | 61.3% | 0.54% | 1.03 | 1.00x |

**A correction to how the first version of this cost model read.** It scored windows by
truncating at the ceiling, which reported 7.5% of speech lost at 10 s and 22.4% for
IndicVoices alone. That is not what the pipeline does: `merge_to_window` splits an
over-long region into back-to-back windows, so a 28 s clip becomes three training clips.
Real loss is the sub-`min_clip_seconds` tail — **0.54%**, not 7.5%. The truncating model
overstated it by more than an order of magnitude, and it overstated it worst on exactly
the long-form sources the window decision is about.

With that corrected, data loss no longer discriminates between the candidates: every
window from 4 s up discards under 2% of speech. What is left is a straight trade between
training compute and task fit, and decision 0004 already rules on task fit.

Note also that shrinking the window saves less than it appears: 10 s → 6 s is 0.67x the
corpus-pass compute, not 0.60x, because the same audio emits 1.14 windows per clip instead
of 1.03.
