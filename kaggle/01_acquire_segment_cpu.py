"""Step 1 -- stream Vaani Hindi, VAD-segment to <=10 s, pack as a private dataset.

    Accelerator: NONE (CPU)     Internet: required      Quota: free

Everything here is CPU work, which is why it must not run in a GPU session. Download,
resample, VAD segmentation and packing off the GPU budget is what makes the plan fit.

PREREQUISITES
  * Accept the terms on huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part
    (CC-BY-4.0, gated, free). A 403 an hour in is avoidable.
  * HF_TOKEN in Add-ons -> Secrets.

WHY STREAMING
  The Hindi config holds 963 h across ~110 GB. We want 5 h for the smoke test and ~120 h
  eventually. `streaming=True` plus an early break pulls only what we consume; materialising
  the config would exhaust the session disk.
"""

import json
import sys
import time
from pathlib import Path

TARGET_HOURS = 5.0          # smoke test; raise to 120 after step 3 passes
OUT = Path("/kaggle/working/vaani_hi_segmented")
DATASET = "ARTPARK-IISc/Vaani-transcription-part"
CONFIG = "Hindi"

sys.path.insert(0, "/kaggle/working/whisper_distill/src")


def main() -> None:
    import soundfile as sf
    from datasets import load_dataset
    from kaggle_secrets import UserSecretsClient

    from whisper_distill.config import DEFAULT
    from whisper_distill.data.segment import Segment, merge_to_window

    token = UserSecretsClient().get_secret("HF_TOKEN")
    audio_cfg, data_cfg = DEFAULT.audio, DEFAULT.data

    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    manifest = (OUT / "manifest.jsonl").open("w", encoding="utf-8")

    print(f"streaming {DATASET}:{CONFIG} for {TARGET_HOURS} h ...", flush=True)
    ds = load_dataset(DATASET, CONFIG, split="train", streaming=True, token=token)

    import torch
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_speech_timestamps = utils[0]

    kept_s = 0.0
    n_clips = 0
    start = time.time()

    for i, row in enumerate(ds):
        if kept_s >= TARGET_HOURS * 3600:
            break

        audio = row["audio"]
        wav, sr = audio["array"], audio["sampling_rate"]
        if sr != audio_cfg.sample_rate:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=audio_cfg.sample_rate)

        tensor = torch.from_numpy(wav).float()
        stamps = get_speech_timestamps(
            tensor, model, sampling_rate=audio_cfg.sample_rate, return_seconds=True
        )
        clips = merge_to_window(
            [Segment(float(s["start"]), float(s["end"])) for s in stamps],
            max_seconds=data_cfg.max_clip_seconds,
            min_seconds=data_cfg.min_clip_seconds,
        )

        for j, seg in enumerate(clips):
            a = int(seg.start_s * audio_cfg.sample_rate)
            b = int(seg.end_s * audio_cfg.sample_rate)
            clip_id = f"vaani_hi_{i:06d}_{j:02d}"
            rel = f"clips/{clip_id}.wav"
            sf.write(OUT / rel, wav[a:b], audio_cfg.sample_rate)
            manifest.write(json.dumps({
                "clip_id": clip_id,
                "audio_path": rel,
                "duration_s": round(seg.duration_s, 3),
                "source": f"{DATASET}:{CONFIG}",
                # The human reference is kept for the WER filter in step 2. It is
                # normalised and unpunctuated -- that is exactly why we pseudo-label
                # anyway rather than training on it.
                "reference": row.get("transcript") or row.get("text") or "",
                "parent_row": i,
                "parent_offset_s": round(seg.start_s, 3),
            }, ensure_ascii=False) + "\n")
            kept_s += seg.duration_s
            n_clips += 1

        if i % 25 == 0:  # heartbeat -- silent cells get killed
            print(f"  row {i:>6}  clips {n_clips:>6}  kept {kept_s / 3600:5.2f} h  "
                  f"elapsed {(time.time() - start) / 60:5.1f} min", flush=True)

    manifest.close()
    print(f"\ndone: {n_clips} clips, {kept_s / 3600:.2f} h in "
          f"{(time.time() - start) / 60:.1f} min")
    print(f"mean clip length: {kept_s / max(n_clips, 1):.2f} s")
    print(f"\nNow: Save Version, then File -> 'Create Dataset from Output' so step 2 can "
          f"mount it read-only at /kaggle/input/")


if __name__ == "__main__":
    main()
