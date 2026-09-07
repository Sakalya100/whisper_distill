# Execution plan — Hinglish dictation distillation on free-tier Kaggle

**Living document.** Last revised 2026-09-08.
Published page: <https://claude.ai/code/artifact/69fb3c75-b71a-4917-b987-2c6b851b32c1>
Source: [`artifacts/kaggle-plan.html`](artifacts/kaggle-plan.html)

Target: a ~132M-parameter, ~135 MB INT8 Hinglish dictation model that runs on a 4 GB
Android phone, distilled from a Hindi Whisper-medium teacher, trained entirely inside
Kaggle's free 30-hour weekly GPU quota.

The plan is ordered by **what has to be measured before the rest of it means anything.**
Two gates up front, then a schedule that survives a session disconnect.

---

## Gate 0 — how does Kaggle bill a T4×2 session?

`[UNVERIFIED]` — settled by `kaggle/00_quota_probe.py`, ~30 minutes of quota.

The 30 hours is one pool shared across P100 and T4×2. Nothing in Kaggle's docs or product
announcements states whether the meter charges **session wall-clock** or **GPU-hours**.
It moves the schedule ~2× and decides which experiments are affordable at all.

### Confirmed on the account, 2026-09-08

Phone verification **Verified**. GPU `00:00 / 30 hrs`, TPU `00:00 / 20 hrs` — a clean zero
baseline, which is the ideal before-value. Private datasets `700.48 MB / 214.75 GB`
(= 200 GiB; the UI just reports it in decimal, so there is no extra headroom to plan for).

**Concurrency cap is at least 2** — two notebooks, both "Version #1 with GPU T4 x2", ran
side by side. That question is settled without a separate experiment.

### Decoding the reading

The probe ran as **two** concurrent T4×2 commits rather than one, so the meter delta has
two candidate readings for `N` sessions of `T` minutes each:

| Meter delta | Verdict | Consequence |
|---|---|---|
| ≈ `N × T` (2 × 15 = **30 min**) | wall-clock billing | A 12 h T4×2 commit costs 12 quota-h and yields ~24 GPU-h. Run everything DDP inside one dual-GPU session; two concurrent *single*-GPU sessions would bill twice for the same work. **≈54 quota-hours total, ~1.8 weeks.** |
| ≈ `N × T × 2` (2 × 15 × 2 = **60 min**) | per-GPU billing | T4×2 buys only VRAM headroom. Cut the ablation block from three runs to one. **≈108 quota-hours total, ~3.6 weeks.** |
| near the **45 min** midpoint | inconclusive | Almost always unequal session runtimes, or startup overhead large relative to a short run. Re-run as a *single* 30-minute commit with nothing else active. |

Two caveats on reading it: both sessions must have run the **same duration** for the
delta to decode, and a committed run pays container startup and teardown on top of the
cell's own runtime — so take each notebook's reported duration from its version history
rather than assuming the sleep length.

`whisper_distill.labeling.quota.decode_quota_probe` implements this and deliberately
refuses to guess at the midpoint, falling back to the pessimistic budget.

### Result of the first probe: inconclusive, and why

Meter went `00:00 → 00:54`. **54 minutes over two concurrent T4×2 sessions is two unknowns
against one equation:**

| Hypothesis | Implied sum of the two session durations |
|---|---|
| wall-clock billing | 54 min (~27 each) |
| per-GPU billing | 27 min (~13.5 each) |

Both are entirely plausible, so the reading does not decide anything on its own. The
missing datum is each notebook's **reported duration from its version history** — if they
summed to ~27 min it is per-GPU; ~54 min and it is wall-clock.

`whisper_distill.labeling.quota.implied_runtimes` does this inverse solve, and
`decode_quota_probe(..., session_minutes=[...])` now takes unequal durations, since
concurrent commits rarely run for the same length.

**The clean re-run:** one T4×2 session, alone, 20 minutes. Wall-clock predicts `+20`,
per-GPU predicts `+40` — a factor of two, no concurrency confound, and startup overhead
cannot bridge the gap. Costs 20–40 min of a 30-hour week. A P100 probe cannot substitute:
with one GPU both hypotheses predict the same delta.

### Confirmed: CPU sessions are free

A CPU-only session left the GPU meter untouched. This was the load-bearing assumption of
the whole workflow, and it holds. Consequences, all real rather than hoped-for:

- Every `0` in the budget table below is genuinely zero, not rounded down.
- **Step 1 is unblocked now**, independent of how Gate 0 resolves — acquisition and
  segmentation cost no quota, so there is no reason to wait on the billing question.
- CPU sessions are also concurrent (≥2, same as GPU) and run 12 h each, so the 200-hour
  acquisition stage — the item flagged as under-costed — can be split across parallel
  sessions by source or by shard range at no cost.

The corollary is a discipline, not a nicety: **anything that does not need a GPU must not
run in a GPU session.** A stray `librosa.load` inside a training notebook is quota spent on
work that was available for free.

### Still open

`[UNVERIFIED]` **Actual scratch disk on a GPU session.** The ~20 GB figure is the *output*
cap. Print `df -h`.

**Blocks:** the schedule, and whether the three-ablation block is affordable.
Do not start data acquisition before the billing verdict is written down.

---

## Gate 1 — can the teacher actually write English?

`[UNVERIFIED]` — ~1 GPU-hour. See [decision 0001](decisions/0001-teacher-checkpoint.md).

`vasista22/whisper-hindi-medium` was fine-tuned on GramVaani, ULCA, Shrutilipi and
FLEURS — four corpora that are entirely Devanagari, normalised and unpunctuated. Two
consequences, both project-blocker class:

1. If the teacher **transliterates** Latin-script English instead of preserving it, every
   pseudo-label is wrong in precisely the dimension a Hinglish dictation model exists to
   serve. No student-side fix recovers that.
2. The pseudo-labelling premise is "the teacher gives you punctuation and casing the human
   labels can't." That assumes the teacher *has* punctuation. A Hindi fine-tune on
   unpunctuated references may have lost it.

### The test

Run the candidate over 30 minutes of genuinely code-mixed speech and read the output.
One utterance decides it:

| Spoken | Pass | Fail |
|---|---|---|
| "meeting chaar baje hai, Slack pe ping karo" | `meeting 4 baje hai, Slack pe ping karo` | `मीटिंग चार बजे है, स्लैक पे पिंग करो` |

Latin-script `Slack` surviving is the signal. The `chaar → 4` conversion is **not** the
model's job — that belongs in rule-based inverse text normalisation downstream
(`src/whisper_distill/itn/`).

**Blocks:** the mel configuration. A failure reopens the teacher to `openai/whisper-large-v3`
or `ARTPARK-IISc/whisper-large-v3-vaani-hindi`, both **128 mel**, which forfeits
single-pass feature extraction. Build the cache pipeline only after this resolves.

The same caveat applies to `vasista22/whisper-hindi-small` as student init. If English is
gone, initialise from `openai/whisper-small` and absorb the extra epochs. Either way:
GramVaani telephone audio is already baked into those weights, so dropping telephone data
from the corpus does not drop it from the model.

---

## Corrections to the original spec

**The mel cache is 80×1000, not 80×500.** Whisper runs at 100 frames/sec (hop 160 at
16 kHz): 30 s is 3000 frames, a 10 s window is 1000. The 500 is the post-convolution
*encoder* length. 160 KB/clip fp16 → **11.5 GB** for 72,000 clips, not 5.8. See
[decision 0002](decisions/0002-mel-config-and-cache-layout.md).

**Whisper's mel padding is not zero.** Whisper normalises log-mel as `(log10(mag) + 4) / 4`
after flooring at `log_spec.max() - 8`, so padded frames hold a negative constant sitting
exactly `2.0` below the clip's normalised maximum. Since the encoder is **frozen** it
cannot adapt to a different convention, which makes zero-padding the mel cache a silent
train/inference mismatch across the whole corpus. Take the teacher's own columns wholesale
(`feats[i, :, :1000]`) rather than re-padding. Caught by review, not by tests — the
original test asserted the padding *was* zero, which is why 62 green tests missed it.

**Vaani's 2,043 hours is the transcribed subset, not the corpus.** The full collection is
31,255 hours of audio. The Hindi config of `ARTPARK-IISc/Vaani-transcription-part` holds
**963 hours**; all 59 language configs total **234 GB**. CC-BY-4.0, gated but free. We want
120 of those 963 hours, so **stream with an early stop** — pulling the whole config will
exhaust the session disk. `ai4bharat/IndicVoices` is CC-BY-4.0 and HF-hosted, fine as
specified.

---

## Workflow — working with the grain of the platform

Nine rules that between them decide whether ~108 GPU-hours of work fits in the quota.

1. **Phone-verify first.** Gates both accelerators and notebook internet. Until it's done
   the GPU dropdown is greyed out.

2. **Push every non-GPU stage into CPU notebooks.** Kaggle meters GPU and TPU; CPU
   notebooks appear not to draw on the 30-hour pool — `[UNVERIFIED]`, confirmed by Gate 0.
   CPU sessions give 2× Xeon, 32 GB RAM, 12 hours. Move download, VAD segmentation,
   resampling, WER filtering, tokenisation, dataset packing, ITN development and all eval
   scoring there. This deletes the "3 GPU hours" prep line outright.

3. **Segment to ≤10 s on CPU *before* labelling.** A prerequisite, not a detail — it is
   what makes rule 4 work. `silero-vad`.

4. **Fuse pseudo-labelling and mel precompute into one GPU pass.** You pay GPU time for
   labelling regardless; mel extraction rides along in dataloader workers. A ≤10 s clip's
   student mel is a *prefix* of the teacher's 30 s padded mel, so one FFT serves both.

5. **Every long run is `Save & Run All (Commit)`.** Never an interactive tab — those
   idle-timeout well short of 12 hours. Print a heartbeat every N steps; long silent cells
   can trip iopub and per-cell timeouts even while the GPU is busy.

6. **Checkpoint to a private HF Hub repo, not Kaggle output.** `/kaggle/working` persists
   between runs but caps around 20 GB and versioning it every 500 steps is clumsy. Use
   `push_to_hub=True`, `hub_strategy="all_checkpoints"`, `hub_private_repo=True`,
   `save_total_limit=1`, then resume from the Hub on session start. `HF_TOKEN` goes in
   **Add-ons → Secrets**, never in the notebook body.

7. **Ship the cache as a private Kaggle dataset — sharded.** 200 GB private quota, 200 GB
   per dataset, **max 50 top-level files**. Pack into memmapped `.npy` shards plus a parquet
   index, not 72,000 loose files.

8. **On T4×2, don't use the Trainer default.** It falls back to naive model parallelism and
   leaves one GPU idle for most of the step. Launch DDP through `torchrun` or `accelerate`.

9. **Never touch Kaggle TPU.** Importing `transformers` Trainer or pipeline classes crashes
   TPU sessions (HF issue #28609). The separate 20 TPU hours/week are not usable here.

**Label with CTranslate2, distil with transformers.** Convert the teacher to CT2 and run
batched `faster-whisper`: ~4× from the backend plus 2–4× from batching, taking the 20-hour
labelling line down to 6–10. The caveat that keeps it honest: **CT2 returns no logits**, so
use it for the bulk cross-entropy labels and keep HF transformers for the 50-hour KL subset
only. `int8_float16` is tuned for Ampere+; T4 is Turing, so benchmark it against plain
`float16` on 30 minutes of audio before committing the full pass.

---

## Budget

Compute required, in **GPU-hours**. What that costs in *quota*-hours is what Gate 0 decides.

| Stage | Where | GPU-hours |
|---|---|---:|
| Acquisition, VAD segmentation, resample, filter, pack | CPU session | 0 |
| Teacher code-switch audit (Gate 1) + 5 h smoke test | T4×2 | 2–3 |
| Fused label + mel + token cache, 200 h, CT2 teacher | T4×2 | 6–10 |
| Top-k logit pass, 50 h subset, HF transformers | T4×2 | 4–6 |
| Main run — frozen encoder, 4 decoder layers, ~4 epochs | T4×2 DDP | 12–18 |
| Three ablations — layer count, vocab size, window length | T4×2 DDP | 30–45 |
| KL ablation training | T4×2 DDP | 8–10 |
| Quantisation, ONNX export, sanity evals | mostly CPU | 2–3 |
| ITN post-processor, eval scoring, false-trigger tests | CPU session | 0 |
| Slack for failed runs and disconnects | — | 25–30 |
| **Total compute** | | **90–125** |

At the 108-hour mid-estimate: **54 quota-hours (1.8 weeks)** under wall-clock billing,
**108 quota-hours (3.6 weeks)** under per-GPU billing. The wall-clock figure is a floor,
not a forecast — only stages that genuinely scale under DDP get the halving. The
twelve-week wall-clock estimate stands either way; data prep, app work and debugging all
happen off-GPU.

### Still under-costed

- **Getting the audio in.** 200 hours is ~12–23 GB compressed and nothing accounts for the
  transfer. Reserve a full CPU session and assume one redo after the first attempt hits a
  disk or rate limit.
- **Disk headroom nobody has printed.** The ~20 GB figure is the *output* cap; actual
  session scratch is larger. Run `df -h` at session start.
- **The 30-hour YouTube scrape.** Run `yt-dlp` locally — Kaggle egress addresses get
  throttled and blocked in practice. Terms-of-service exposure is the project owner's to
  accept; keep the scrape unpublished, or restrict to Creative-Commons uploads to retain
  the option of releasing the corpus.

---

## Model plan

| Change | Params after |
|---|---:|
| whisper-small Hindi baseline | ~244 M |
| Decoder 12 → 4 layers (maximally spaced copy) | ~160 M |
| Vocabulary pruned to ~16 k | ~132 M |
| 10 s window (positional embeddings sliced) | ~132 M, ~3× less encoder compute |

INT8 → roughly 135 MB.

**Freeze the encoder.** No gradients or optimiser state for 88 M parameters roughly doubles
usable batch size on a T4, and keeping the full encoder preserves robustness across audio
conditions. Encoder shrinking is an end-of-project ablation if quota remains, not a
first-run decision.

**Vocabulary pruning has a trap.** A KL target computed over the teacher's full vocabulary
against a pruned student vocabulary trains fine and produces garbage. Teacher logits must
be indexed to the kept token ids and renormalised first. Implemented once in
`training/distill_loss.py` and asserted in tests — see
[decision 0003](decisions/0003-ce-first-kl-as-ablation.md).

**Note the 30 s → 10 s risk.** Whisper was only ever trained on 30-second zero-padded
input. Slicing the encoder's positional embeddings to 1000 frames is architecturally clean
but off-distribution; expect degradation that fine-tuning has to recover. This is why
window length is one of the three ablations rather than an assumption.

---

## Data

Target ~200 hours. At this compute, quota is the bottleneck, not data volume.

| Source | Hours | Notes |
|---|---:|---|
| `ARTPARK-IISc/Vaani-transcription-part` (Hindi) | ~120 | 963 h available, CC-BY-4.0, gated. Stream with early stop. |
| `ai4bharat/IndicVoices` (Hindi) | ~50 | Spontaneous/extempore — closest to what dictation sounds like. CC-BY-4.0, gated. |
| Scraped Hinglish (tech reviews, podcasts, vlogs) | ~30 | Where dense natural code-switching lives. No dataset provides this. |

**Pseudo-label all of it**, including the parts with human transcripts, so label style is
uniform. Filter by teacher WER against the human reference where one exists and drop above
~20%. For scraped audio use teacher entropy as the proxy filter.

Augment with SpecAugment plus light room reverb and phone-mic noise. **No codec
simulation** — telephone audio was a call-centre concern; dictation is close-mic, single
speaker, moderate noise. The acoustic problem is narrower than the original spec assumed,
which is exactly what a small compute budget wants.

---

## The eval set is the deliverable

Record it. 2–3 hours, 15–20 speakers, their own phones, note-style Hinglish: reminders,
shopping lists, meeting notes, voice messages. Mix quiet rooms with a fan running and a TV
on. Include numbers, times and dates deliberately.

Vistaar and Kathbath are read speech and will not tell you whether a dictation model is
good. A purpose-built Hinglish dictation eval set does not appear to exist publicly, so
building one is a contribution independent of whether the model works. Spec in
[`../eval_set/README.md`](../eval_set/README.md).

Report four things:

1. WER and CER, split by code-mixing density
2. **Punctuated** WER, not only normalised WER
3. Entity accuracy on numbers, dates and times
4. On-device RTF, peak RAM and battery drain per hour on one real mid-range Android

Baselines: the teacher, `whisper-hindi-small` INT8 from the sherpa-onnx Indic packs,
Distil-Large-v3, Moonshine.

### Two things that live outside the model

**Inverse text normalisation as a rule-based post-processor.** Do not make a 132M model
learn that "saade chaar baje" is "4:30". ~200 lines of Python improves perceived quality
more than three points of WER. `src/whisper_distill/itn/`.

**Measure the false-trigger rate on silence and pure noise.** Whisper hallucinates during
silence and distillation tends to make it worse. For a dictation app that leaves the mic
open, this is the failure users will actually report. Test it explicitly and put a VAD in
front of the model. `src/whisper_distill/evaluation/false_trigger.py`.

---

## First moves

Each step is cheap and each one can invalidate the next. If step 3 fails on 5 hours it will
fail on 200, and you will have spent one quota-hour finding out instead of forty.

| # | Step | Cost |
|---|---|---|
| 0 | Phone-verify, run the meter experiment. Produces the billing answer, the concurrency cap, and the CPU-is-free confirmation. | ~30 min quota |
| 1 | **CPU session:** stream Vaani Hindi, take 5 hours, VAD-segment to ≤10 s, push as a private Kaggle dataset. | 0 quota |
| 2 | **GPU:** audit the teacher on code-mixed audio *first* — it decides teacher and mel config. Then label the 5 hours, dumping mels and tokens in the same pass. Eyeball 50. Settle the script convention here. | ~1 h |
| 3 | **GPU:** cut whisper-small to 4 decoder layers, train on the 5 hours, confirm loss drops *and* English words survive. Push to a private Hub repo — this also proves checkpoint-and-resume before you need it. | ~1 h |
| 4 | Only then scale to 200 hours. | — |

---

## Sources

- [AI4Bharat / Vistaar](https://github.com/AI4Bharat/vistaar) · [issue #4 — loading IndicWhisper](https://github.com/AI4Bharat/vistaar/issues/4)
- [`vasista22/whisper-hindi-medium`](https://huggingface.co/vasista22/whisper-hindi-medium) · [`whisper-hindi-small`](https://huggingface.co/vasista22/whisper-hindi-small) · [`parthiv11/indic_whisper_nodcil`](https://huggingface.co/parthiv11/indic_whisper_nodcil)
- [`ARTPARK-IISc/Vaani-transcription-part`](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part) · [`ai4bharat/IndicVoices`](https://huggingface.co/datasets/ai4bharat/IndicVoices) · [`whisper-large-v3-vaani-hindi`](https://huggingface.co/ARTPARK-IISc/whisper-large-v3-vaani-hindi)
- [Distil-Whisper training README](https://github.com/huggingface/distil-whisper/blob/main/training/README.md) · [Distil-Whisper paper](https://arxiv.org/abs/2311.00430)
- [fine-tune-whisper-vi — Kaggle gotchas](https://github.com/phineas-pta/fine-tune-whisper-vi/blob/main/README.md) · [vasistalodagala/whisper-finetune](https://github.com/vasistalodagala/whisper-finetune)
- [HF × Kaggle integration](https://huggingface.co/blog/kaggle-integration) · [Kaggle floating GPU quota](https://www.kaggle.com/product-feedback/173129) · [Kaggle private dataset quota](https://www.kaggle.com/product-announcements/512322)
- [Batched Whisper speedups](https://mobiusml.github.io/batched_whisper_blog/) · [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
