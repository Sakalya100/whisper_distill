# 0003 — Cross-entropy first, KL divergence as a controlled ablation

**Date:** 2026-09-08
**Status:** accepted

## Decision

The first full training run is **shrink-and-fine-tune**: copy maximally spaced decoder
layers from the initialised student, then train on cross-entropy against teacher
pseudo-labels only. No KL term.

This is a named variant in the Distil-Whisper training documentation, not an improvisation.

## Why, on this budget

Full KL distillation needs teacher logits at every step. Either run the teacher in the
training loop — roughly halving throughput on a quota-metered GPU — or precompute top-k
logits, which is a separate pass plus gigabytes of disk. Neither is affordable before a
working model exists.

## The ablation

Add KL on a 50-hour subset afterwards, as a controlled comparison. Budgeted at 4–6
GPU-hours for the top-k logit pass plus 8–10 for training. **The result is worth writing up
either way**: if KL barely helps at 200 hours of data, that is a finding.

## The trap to not walk into

The student's vocabulary is pruned from ~51.8k to ~16k. A KL target computed over the
teacher's full vocabulary against the student's pruned vocabulary is silently wrong — it
will train, and produce garbage. The teacher logits must be **indexed down to the kept
token ids and renormalised** before the KL is computed.

This is implemented once, in `training/distill_loss.py`, and asserted in tests rather than
left as a comment.
