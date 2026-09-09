"""Step 1 -- stream Vaani Hindi, VAD-segment to <=10 s, write a manifest.

    Accelerator: NONE (CPU)     Internet: ON     Quota: FREE (measured -- see docs)

Everything here is CPU work. Confirmed 2026-09-08 that CPU sessions do not draw on the
30-hour GPU pool, so this stage is genuinely free and does not wait on the Gate 0
billing question.

PREREQUISITES
  1. Accept the terms on huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part
     (CC-BY-4.0, gated, free).
  2. HF_TOKEN in Add-ons -> Secrets. Never in the notebook body.
  3. Settings -> Accelerator: None. Internet: On. Persistence: Files only.

WHY STREAMING
  The Hindi config holds 963.04 h. We want 5 h now and ~120 h later. streaming=True with
  an early break pulls only what we consume; materialising the config would exhaust the
  session disk long before it finished.

WHY decode=False
  datasets 4.0 changed the Audio feature to return a torchcodec AudioDecoder, which needs
  a matching torch build and a system FFmpeg. Casting to Audio(decode=False) hands us raw
  bytes instead and we decode with soundfile, which is stable across versions.
  data/audio_io.py handles every shape anyway, so this is belt and braces.

SCALE-UP NOTE
  At 5 h this writes ~600 MB of FLAC, comfortably inside the ~20 GB notebook output cap.
  At 120 h it would be ~14 GB -- still inside, but close. For the full run, split across
  several CPU sessions by SKIP_ROWS/TARGET_HOURS and publish each as its own dataset
  version. Do NOT write WAV at that scale; FLAC is lossless and roughly half the size.
"""

import json
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- configure
TARGET_HOURS = 5.0     # smoke test. Raise only after step 3 passes.
SKIP_ROWS = 0          # for splitting the 120 h run across sessions
DATASET = "ARTPARK-IISc/Vaani-transcription-part"
CONFIG = "Hindi"
SPLIT = "train"        # Vaani exposes train / validation / test
OUT = Path("/kaggle/working/vaani_hi_segmented")
REPO_SRC = "/kaggle/working/whisper_distill/src"  # /kaggle/tmp/... keeps it out of the output
# --------------------------------------------------------------------------------------

sys.path.insert(0, REPO_SRC)


def session_facts() -> None:
    """Print the disk numbers nobody has looked at yet. Costs nothing, ends a guess."""
    import subprocess

    for cmd in (["df", "-h", "/kaggle/working"], ["nproc"], ["free", "-g"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
            print(f"$ {' '.join(cmd)}\n{out.rstrip()}\n")
        except Exception as e:  # noqa: BLE001 - diagnostic only
            print(f"$ {' '.join(cmd)} -> {e}\n")

    import datasets
    print(f"datasets {datasets.__version__}")
    try:
        import soundfile
        print(f"soundfile {soundfile.__version__}")
    except ImportError:
        print("soundfile MISSING -- pip install soundfile before streaming")


def already_done(manifest_path: Path) -> tuple[set[int], float, int]:
    """Resume support: which parent rows are already segmented, and how much we have.

    /kaggle/working persists between runs of the same notebook, so a session that dies
    two hours in should not start from zero. Returns (done_rows, kept_seconds, n_clips).
    """
    if not manifest_path.exists():
        return set(), 0.0, 0
    rows: set[int] = set()
    kept = 0.0
    n = 0
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line from a killed session
            rows.add(rec["parent_row"])
            kept += rec["duration_s"]
            n += 1
    if rows:
        print(f"resuming: {n} clips / {kept / 3600:.2f} h already done, "
              f"{len(rows)} parent rows to skip\n")
    return rows, kept, n


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Audio, load_dataset
    from kaggle_secrets import UserSecretsClient

    from whisper_distill.config import DEFAULT
    from whisper_distill.data.audio_io import decode_audio_field, probe_schema
    from whisper_distill.data.segment import Segment, merge_to_window

    audio_cfg, data_cfg = DEFAULT.audio, DEFAULT.data
    token = UserSecretsClient().get_secret("HF_TOKEN")

    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.jsonl"
    done_rows, kept_s, n_clips = already_done(manifest_path)

    # ---------------------------------------------------------------- schema probe first
    # Pull ONE row and assert the columns we depend on. A schema mismatch should cost ten
    # seconds, not an hour -- and an empty `transcript` silently disables step 2's WER
    # filter, which is worse than a crash because it looks like it worked.
    print(f"probing {DATASET}:{CONFIG}:{SPLIT} ...", flush=True)
    probe = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True, token=token)
    first = next(iter(probe))
    print(f"columns     : {sorted(first)}")
    print(f"audio field : {probe_schema(first)}")
    print(f"transcript  : {first.get('transcript', '')[:90]!r}")
    if not (first.get("transcript") or "").strip():
        print("\nWARNING: first row's transcript is empty. If that holds across rows, the "
              "WER filter in step 2 has nothing to filter against.\n")
    del probe, first

    # ------------------------------------------------------------------------- the stream
    ds = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True, token=token)
    ds = ds.cast_column("audio", Audio(decode=False))  # bypass torchcodec; see docstring

    print("loading silero-vad ...", flush=True)
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_speech_timestamps = utils[0]

    manifest = manifest_path.open("a", encoding="utf-8")
    target_s = TARGET_HOURS * 3600
    start = time.time()
    decode_paths: dict[str, int] = {}
    skipped_no_speech = 0
    row_i = -1

    print(f"\nstreaming for {TARGET_HOURS} h of speech ...\n", flush=True)
    try:
        for row_i, row in enumerate(ds):
            if kept_s >= target_s:
                break
            if row_i < SKIP_ROWS or row_i in done_rows:
                continue

            try:
                wav, how = decode_audio_field(row["audio"], target_sr=audio_cfg.sample_rate)
            except Exception as e:  # noqa: BLE001 - one bad file must not kill the run
                print(f"  row {row_i}: decode failed ({type(e).__name__}: {e}); skipping")
                continue
            decode_paths[how] = decode_paths.get(how, 0) + 1

            stamps = get_speech_timestamps(
                torch.from_numpy(np.ascontiguousarray(wav)),
                model,
                sampling_rate=audio_cfg.sample_rate,
                return_seconds=True,
            )
            clips = merge_to_window(
                [Segment(float(s["start"]), float(s["end"])) for s in stamps],
                max_seconds=data_cfg.max_clip_seconds,
                min_seconds=data_cfg.min_clip_seconds,
            )
            if not clips:
                skipped_no_speech += 1
                continue

            for j, seg in enumerate(clips):
                a = int(seg.start_s * audio_cfg.sample_rate)
                b = int(seg.end_s * audio_cfg.sample_rate)
                clip_id = f"vaani_hi_{row_i:06d}_{j:02d}"
                rel = f"clips/{clip_id}.flac"
                # FLAC: lossless, ~half of WAV. At 120 h the difference decides whether
                # the output fits the ~20 GB cap.
                sf.write(OUT / rel, wav[a:b], audio_cfg.sample_rate, format="FLAC")
                manifest.write(json.dumps({
                    "clip_id": clip_id,
                    "audio_path": rel,
                    "duration_s": round(seg.duration_s, 3),
                    "source": f"{DATASET}:{CONFIG}",
                    # Normalised and unpunctuated -- kept for step 2's WER filter, NOT as
                    # a training target. That style is exactly why we pseudo-label.
                    "reference": row.get("transcript") or "",
                    "language": row.get("language", ""),
                    "gender": row.get("gender", ""),
                    "district": row.get("district", ""),
                    "parent_row": row_i,
                    "parent_offset_s": round(seg.start_s, 3),
                }, ensure_ascii=False) + "\n")
                kept_s += seg.duration_s
                n_clips += 1

            if row_i % 25 == 0:  # heartbeat -- silent cells get killed
                manifest.flush()
                el = (time.time() - start) / 60
                rate = kept_s / 3600 / max(el, 1e-9)
                print(f"  row {row_i:>6}  clips {n_clips:>6}  kept {kept_s / 3600:5.2f} h  "
                      f"{el:5.1f} min  {rate:4.1f} h/min  "
                      f"eta {(target_s - kept_s) / 3600 / max(rate, 1e-9):5.1f} min",
                      flush=True)
    finally:
        manifest.flush()
        manifest.close()

    # ------------------------------------------------------------------------- the report
    mean_clip = kept_s / max(n_clips, 1)
    print("\n" + "=" * 66)
    print(f"clips            : {n_clips}")
    print(f"speech kept      : {kept_s / 3600:.2f} h")
    print(f"mean clip length : {mean_clip:.2f} s   (window is {data_cfg.max_clip_seconds:.0f} s)")
    print(f"rows consumed    : {row_i + 1}")
    print(f"rows w/o speech  : {skipped_no_speech}")
    print(f"decode paths     : {decode_paths}")
    print(f"wall clock       : {(time.time() - start) / 60:.1f} min")
    print("=" * 66)

    # The one number worth reacting to. If VAD is shredding recordings into 2-second
    # pieces, the 10 s window is mostly padding and the corpus is far more clips than it
    # needs to be -- which costs GPU time in step 2 for no extra audio.
    if mean_clip < 4.0:
        print(
            f"\nWARNING: mean clip is only {mean_clip:.2f} s. VAD is over-fragmenting.\n"
            "  Raise max_gap_s in merge_to_window (0.6 -> 1.0) and re-run. Short clips\n"
            "  waste the 10 s window on padding and inflate the clip count, which costs\n"
            "  step-2 GPU time without adding audio."
        )
    elif mean_clip > 9.5:
        print(
            f"\nNOTE: mean clip is {mean_clip:.2f} s, near the ceiling. Recordings are\n"
            "  being split mid-utterance rather than at pauses. Acceptable, but expect\n"
            "  some clips to start or end mid-word."
        )
    else:
        print(f"\nmean clip length looks healthy for a {data_cfg.max_clip_seconds:.0f} s window.")

    print("\nNEXT: Save Version -> then on the notebook page, Output -> "
          "'Create Dataset from Output'. Note the dataset slug and put it in\n"
          "      kaggle/02_audit_and_label_gpu.py as IN.")


if __name__ == "__main__":
    session_facts()
    main()
