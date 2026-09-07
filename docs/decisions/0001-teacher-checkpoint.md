# 0001 — Teacher is `vasista22/whisper-hindi-medium`, not IndicWhisper

**Date:** 2026-09-08
**Status:** accepted, pending the Gate 1 audit

## Context

The plan called for IndicWhisper medium as teacher, on the strength of its Vistaar
benchmark results (lowest WER on 39 of 59 tasks) and its 80-mel feature config, which
matches whisper-small and lets one feature-extraction pass serve teacher and student.

## What we found

IndicWhisper is **not distributed as a Hugging Face repo.** The Vistaar repository links
Hindi weights as `hindi_models.zip` on e2enetworks object storage, and
[issue #4](https://github.com/AI4Bharat/vistaar/issues/4) — a request for instructions on
loading the model in Python at all — has been open since April 2024 with no resolution.
Loading it is unbudgeted, format-unknown work sitting on the critical path of week 1.

## Decision

Use **`vasista22/whisper-hindi-medium`**:

- whisper-medium base, so **80 mel bins** — the shared-extraction decision survives
- plain HF checkpoint, loads via `AutoModelForSpeechSeq2Seq` in one line
- Speech Lab, IIT Madras; funded under Bhashini
- 6.82 WER on Google/FLEURS, 11.38 on Common Voice 11.0

Fallback if the official weights become necessary: `parthiv11/indic_whisper_nodcil`, a
community HF conversion of IndicWhisper. Provenance unverified; not preferred.

## What would reopen this

The Gate 1 audit (see the plan). That checkpoint was fine-tuned on GramVaani, ULCA,
Shrutilipi and FLEURS — all Devanagari, normalised, unpunctuated. If it transliterates
Latin-script English rather than preserving it, **every pseudo-label is wrong in the one
dimension this project exists to serve**, and the teacher reopens to `openai/whisper-large-v3`
or `ARTPARK-IISc/whisper-large-v3-vaani-hindi`.

Both of those are **128 mel**, which forfeits single-pass feature extraction and forces a
separate student mel stage. So decision 0002 is downstream of this audit and must not be
built before it resolves.
