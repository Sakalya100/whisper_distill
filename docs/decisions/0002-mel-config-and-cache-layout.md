# 0002 — 80×1000 mel cache, sharded memmap + parquet index

**Date:** 2026-09-08
**Status:** accepted, conditional on 0001

## The arithmetic that was wrong

The original plan sized the cache at 80×500 per 10-second clip. Whisper runs at
**100 frames per second** (hop 160 at 16 kHz), so:

| Window | Mel frames | Post-conv encoder positions |
|---|---|---|
| 30 s (Whisper native) | 3000 | 1500 |
| 10 s (our student) | **1000** | 500 |

The 500 is the *post-convolution encoder* length, one stage later than the cache. Correct
per-clip size is `80 × 1000 × 2 bytes = 160 KB` in fp16, so 72,000 clips is **11.5 GB**,
not 5.8. Comfortable inside Kaggle's 200 GB private-dataset quota — but the dataset must
not be provisioned for the smaller figure.

## The prefix property

A ≤10 s clip's student mel (80×1000) is a **prefix** of the teacher's 30 s padded mel
(80×3000). One FFT therefore serves both models, which is what makes the fused
label-and-cache pass possible. Two conditions:

1. Clips must be segmented to ≤10 s *before* labelling — hence VAD segmentation is a
   CPU-stage prerequisite, not an optimisation.
2. **The prefix must be taken with the teacher's own padding, not re-padded.** Whisper
   normalises log-mel as `(log10(mag) + 4) / 4` after flooring at `log_spec.max() - 8`, so
   its padded frames hold a negative constant sitting exactly `2.0` below the clip's
   normalised maximum — not `0.0`. Since the encoder is **frozen**, it cannot adapt to a
   different padding convention, so zero-padding would be a silent train/inference
   mismatch across the whole corpus. `ShardWriter` therefore prefers a full-width mel and
   falls back to `whisper_floor(mel)` rather than zeros.

## Storage layout

Kaggle datasets cap at **50 top-level files**, so 72,000 loose `.npy` files is not an
option. Layout:

```
shard-0000.mels.npy      # (N, 80, 1000) fp16 memmap
shard-0000.tokens.npy    # (N, L) int32, right-padded with -100
index.json               # clip_id, shard, row, n_frames, n_tokens, source, teacher_wer, entropy
```

~4 GB per shard, three shards, one index. JSON rather than parquet: the index is a few MB
of metadata read once at startup, so a columnar format buys nothing and adds a dependency. Read memory-mapped from `/kaggle/input`,
which turns the dataloader from a CPU bottleneck into a page-cache read.

## What would reopen this

A failed Gate 1 audit forcing a 128-mel teacher. The student cache stays 80×1000, but the
teacher then needs its own 128×3000 pass, roughly doubling the labelling stage and removing
the prefix property.
