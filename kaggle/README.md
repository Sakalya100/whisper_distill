# Kaggle notebooks

Paste-ready scripts, one per pipeline stage. Each header states which accelerator it wants
and whether it burns quota.

| Script | Accelerator | Quota | Stage |
|---|---|---|---|
| `00_quota_probe.py` | GPU T4×2 | ~15–30 min | Gate 0 — how does the meter bill? |
| `01_acquire_segment_cpu.py` | **None (CPU)** | free | Stream Vaani, VAD-segment to ≤10 s, pack |
| `02_audit_and_label_gpu.py` | GPU T4×2 | ~1 h for 5 h audio | Gate 1 audit, then fused label + mel pass |
| `03_smoke_train_gpu.py` | GPU T4×2 | ~1 h | Cut student, train on 5 h, prove resume works |

## Rules that apply to every one of them

1. **Run as `Save & Run All (Commit)`, never in the interactive tab.** Interactive
   sessions idle-timeout well short of 12 hours. A commit runs headless — close the laptop.
2. **Emit a heartbeat.** A cell that prints nothing for a long stretch can be killed on an
   iopub or per-cell timeout even while the GPU is busy. Every loop here prints progress.
3. **`HF_TOKEN` goes in Add-ons → Secrets**, never in the notebook body. Read it with
   `UserSecretsClient`. Same for `KAGGLE_KEY` if a script uploads a dataset.
4. **Accept the dataset terms first.** Vaani and IndicVoices are gated: open the dataset
   page on huggingface.co, accept, *then* run. A 403 four hours into a run is avoidable.
5. **Print `df -h` at session start.** The ~20 GB figure is the *output* cap, not the
   scratch disk. Don't plan around a number nobody has looked at.
6. **Never select TPU.** Importing `transformers` Trainer or pipeline classes crashes TPU
   sessions. The separate 20 TPU h/week are not usable for this stack.

## Installing the package inside a notebook

```python
!git clone -q https://github.com/Sakalya100/whisper_distill.git /kaggle/working/whisper_distill
%pip install -q -e /kaggle/working/whisper_distill
```

Or, without cloning, add the repo as a Kaggle dataset and `sys.path.insert(0, ".../src")`.
