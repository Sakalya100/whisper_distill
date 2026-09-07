# Dictation eval set

The project deliverable most likely to outlast the model. Vistaar and Kathbath are read
speech; neither tells you whether a dictation model is good. A purpose-built Hinglish
dictation eval set does not appear to exist publicly.

**Recordings are never committed.** `.gitignore` blocks audio; only manifests live here.

## Spec

| Dimension | Target |
|---|---|
| Duration | 2–3 hours |
| Speakers | 15–20, on their own phones |
| Register | note-style Hinglish: reminders, shopping lists, meeting notes, voice messages |
| Environments | quiet room / fan running / TV on — roughly balanced |
| Deliberate content | numbers, times, dates in every session |

## Manifest schema — `manifest.jsonl`

```json
{
  "clip_id": "spk03_0007",
  "speaker_id": "spk03",
  "audio_path": "raw/spk03/0007.wav",
  "duration_s": 6.4,
  "device": "Redmi Note 12",
  "environment": "fan",
  "prompt_type": "reminder",
  "reference": "meeting 4 baje hai, Slack pe ping karo",
  "reference_normalised": "meeting chaar baje hai slack pe ping karo",
  "code_mix_bucket": "dense",
  "entities": [
    {"type": "time", "surface": "4 baje", "value": "16:00"}
  ]
}
```

`code_mix_bucket` is one of `hindi_dominant | balanced | dense` — computed by
`whisper_distill.evaluation.metrics.code_mix_density`, then hand-checked. Results are
reported split by this bucket, because an average WER hides the code-switching failure.

`entities` is what gates perceived quality. A user judges dictation on whether "chaar baje"
became "4 baje", not on average WER. Scored by
`whisper_distill.evaluation.entities`.

## Consent

Speakers are contributing identifiable voice recordings. Collect written consent covering
research use and, separately, whether they permit public release. Store consent status in
the manifest before recording anything you might want to publish later.
