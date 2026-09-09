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

**If the repo is public** — one line, nothing to configure:

```python
!git clone -q https://github.com/Sakalya100/whisper_distill.git /kaggle/working/whisper_distill
```

**If the repo is private**, a plain HTTPS clone stops at `Username for 'https://github.com':`
and the cell hangs forever waiting on stdin that a notebook never provides. Put a GitHub
PAT (scope: `repo`, read is enough) in **Add-ons → Secrets** as `GH_TOKEN` and build the
URL in Python, so the token never appears in a cell, in the output, or in the saved
notebook:

```python
from kaggle_secrets import UserSecretsClient
import subprocess

tok = UserSecretsClient().get_secret("GH_TOKEN")
subprocess.run(
    ["git", "clone", "-q",
     f"https://{tok}@github.com/Sakalya100/whisper_distill.git",
     "/kaggle/working/whisper_distill"],
    check=True,
)
```

Never write the token into a `!git clone` shell line — Kaggle saves cell source *and*
output with the notebook version, so a token pasted there is committed to the notebook and
visible to anyone the notebook is shared with.

Then, in either case:

```python
import sys
sys.path.insert(0, "/kaggle/working/whisper_distill/src")
```

`sys.path.insert` rather than `pip install -e`: the editable install adds nothing here and
costs a dependency-resolution round trip at the start of every session.

**Caveat if you clone into `/kaggle/working`:** the repo becomes part of the notebook's
saved output on every commit, eating into the ~20 GB cap and cluttering the dataset you
publish from it. Clone into `/kaggle/tmp` instead for the stages that publish a dataset,
and adjust `REPO_SRC` at the top of the script to match.
