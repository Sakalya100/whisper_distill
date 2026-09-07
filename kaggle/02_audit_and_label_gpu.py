"""Step 2 -- Gate 1 audit first, then the fused label + mel pass.

    Accelerator: GPU T4 x2      Internet: required      Quota: ~1 h for 5 h of audio

TWO PHASES, AND THE ORDER MATTERS.

Phase A is the Gate 1 audit. `vasista22/whisper-hindi-medium` was fine-tuned on GramVaani,
ULCA, Shrutilipi and FLEURS -- four corpora that are entirely Devanagari, normalised and
unpunctuated. If it transliterates Latin-script English rather than preserving it, every
pseudo-label is wrong in the one dimension this project exists to serve, and no
student-side fix recovers it. **Read the output yourself. Do not skip to phase B.**

    "meeting chaar baje hai, Slack pe ping karo"
      PASS -> meeting 4 baje hai, Slack pe ping karo      (Latin `Slack` survives)
      FAIL -> मीटिंग चार बजे है, स्लैक पे पिंग करो          (transliterated)

A failure reopens the teacher to whisper-large-v3 or whisper-large-v3-vaani-hindi -- both
128 mel, which forfeits single-pass feature extraction. See
docs/decisions/0001-teacher-checkpoint.md.

Phase B is the fused pass. We pay GPU time for labelling regardless, so mel extraction
rides along: a <=10 s clip's student mel (80x1000) is a prefix of the teacher's 30 s padded
mel (80x3000), so one FFT serves both models.
"""

import json
import sys
import time
from pathlib import Path

IN = Path("/kaggle/input/vaani-hi-segmented")   # rename to your dataset slug
OUT = Path("/kaggle/working/mel_cache")
AUDIT_N = 50            # clips to print for phase A
BATCH = 16              # greedy only; beam search spikes VRAM and OOMs on a T4

sys.path.insert(0, "/kaggle/working/whisper_distill/src")


def load_manifest() -> list[dict]:
    with (IN / "manifest.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ======================================================================= PHASE A
def audit(rows: list[dict], n: int = AUDIT_N) -> None:
    """Print teacher output on the most code-mixed clips available. Read it yourself."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from whisper_distill.config import DEFAULT
    from whisper_distill.evaluation.metrics import code_mix_bucket

    cfg = DEFAULT.teacher
    print(f"loading teacher {cfg.model_id} ...", flush=True)
    proc = WhisperProcessor.from_pretrained(cfg.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model_id, torch_dtype=torch.float16
    ).to("cuda").eval()

    # Confirm the mel assumption before trusting any cached feature.
    n_mels = model.config.num_mel_bins
    DEFAULT.audio.assert_teacher_compatible(n_mels)
    print(f"teacher mel bins: {n_mels} -- matches student cache\n")

    # Prefer clips whose human reference already looks code-mixed; that is where the
    # failure mode lives. Falls back to arbitrary clips if none are flagged.
    ranked = sorted(
        rows,
        key=lambda r: (code_mix_bucket(r.get("reference", "")) != "dense",
                       code_mix_bucket(r.get("reference", "")) != "balanced"),
    )[:n]

    import soundfile as sf
    print("=" * 78)
    print("GATE 1 AUDIT -- does Latin-script English survive? does punctuation appear?")
    print("=" * 78)
    latin_survived = punctuated = 0

    for r in ranked:
        wav, _ = sf.read(IN / r["audio_path"], dtype="float32")
        feats = proc(wav, sampling_rate=DEFAULT.audio.sample_rate,
                     return_tensors="pt").input_features.to("cuda", torch.float16)
        ids = model.generate(feats, language=cfg.language, task=cfg.task,
                            num_beams=cfg.num_beams, max_new_tokens=200)
        hyp = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

        has_latin = any("a" <= c.lower() <= "z" for c in hyp)
        has_punct = any(c in ",.?!" for c in hyp)
        latin_survived += has_latin
        punctuated += has_punct

        print(f"\n[{r['clip_id']}]  latin={'Y' if has_latin else 'N'}  "
              f"punct={'Y' if has_punct else 'N'}")
        print(f"  ref : {r.get('reference', '')[:110]}")
        print(f"  hyp : {hyp[:110]}")

    print("\n" + "=" * 78)
    print(f"Latin script present in {latin_survived}/{len(ranked)} outputs")
    print(f"Punctuation present in  {punctuated}/{len(ranked)} outputs")
    print("=" * 78)
    print(
        "\nJudge it yourself -- these counters are a summary, not the verdict.\n"
        "  Latin near zero on code-mixed input => teacher transliterates => STOP.\n"
        "     Switch to a 128-mel teacher and update decisions/0001 and 0002.\n"
        "  Punctuation near zero => the pseudo-labelling premise is weaker than assumed.\n"
        "     Still workable (ITN and casing can be partly rule-based), but say so in the\n"
        "     writeup rather than claiming the teacher donated punctuation."
    )
    del model
    torch.cuda.empty_cache()


# ======================================================================= PHASE B
def label_and_cache(rows: list[dict]) -> None:
    """Fused pass: one mel computation feeds the teacher and the student cache."""
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from whisper_distill.config import DEFAULT
    from whisper_distill.data.pack import ShardWriter
    from whisper_distill.evaluation.metrics import wer
    from whisper_distill.labeling.filters import mean_token_entropy

    cfg, audio_cfg, data_cfg = DEFAULT.teacher, DEFAULT.audio, DEFAULT.data
    proc = WhisperProcessor.from_pretrained(cfg.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model_id, torch_dtype=torch.float16
    ).to("cuda").eval()

    kept = dropped = 0
    start = time.time()

    with ShardWriter(OUT) as writer:
        for b0 in range(0, len(rows), BATCH):
            batch = rows[b0 : b0 + BATCH]
            wavs = [sf.read(IN / r["audio_path"], dtype="float32")[0] for r in batch]

            # ONE feature extraction. `input_features` is the teacher's 80x3000 padded
            # mel; the student's 80x1000 cache is its leading prefix, because every clip
            # is already <=10 s from step 1.
            enc = proc(wavs, sampling_rate=audio_cfg.sample_rate, return_tensors="pt")
            feats = enc.input_features
            teacher_feats = feats.to("cuda", torch.float16)

            with torch.no_grad():
                out = model.generate(
                    teacher_feats,
                    language=cfg.language,
                    task=cfg.task,
                    num_beams=cfg.num_beams,
                    max_new_tokens=200,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            texts = proc.batch_decode(out.sequences, skip_special_tokens=True)

            # Per-clip mean token entropy, the proxy filter for unreferenced audio.
            entropies = []
            for i in range(len(batch)):
                per_step = []
                for step_scores in out.scores:
                    lp = torch.log_softmax(step_scores[i].float(), dim=-1)
                    top = torch.topk(lp, k=min(16, lp.numel())).values
                    per_step.append(top.tolist())
                entropies.append(mean_token_entropy(per_step))

            for i, r in enumerate(batch):
                text = texts[i].strip()
                ref = r.get("reference", "")
                teacher_wer = None
                if ref:
                    e, n = wer(ref, text)
                    teacher_wer = e / n if n else None
                    # Distil-Whisper's WER filter: discard where the teacher
                    # mis-transcribed or hallucinated.
                    if teacher_wer is not None and teacher_wer > data_cfg.max_teacher_wer:
                        dropped += 1
                        continue

                n_frames = min(
                    audio_cfg.n_frames,
                    int(round(r["duration_s"] * 100)),  # 100 mel frames per second
                )
                mel = feats[i, :, :n_frames].numpy()
                token_ids = proc.tokenizer(text, add_special_tokens=True).input_ids

                writer.add(
                    r["clip_id"], mel, token_ids,
                    source=r.get("source", ""),
                    duration_s=r["duration_s"],
                    text=text,
                    teacher_wer=teacher_wer,
                    mean_entropy=entropies[i],
                )
                kept += 1

            if b0 % (BATCH * 10) == 0:  # heartbeat
                done = b0 + len(batch)
                rate = done / max(time.time() - start, 1e-6)
                print(f"  {done:>6}/{len(rows)} clips  kept {kept} dropped {dropped}  "
                      f"{rate:5.1f} clip/s  eta {(len(rows) - done) / max(rate, 1e-6) / 60:5.1f} min",
                      flush=True)

    print(f"\ndone in {(time.time() - start) / 60:.1f} min: kept {kept}, "
          f"dropped {dropped} on the WER filter "
          f"({dropped / max(kept + dropped, 1) * 100:.1f}%)")
    print(f"cache at {OUT} -- Save Version, then Create Dataset from Output")


if __name__ == "__main__":
    rows = load_manifest()
    print(f"{len(rows)} clips in manifest\n")
    audit(rows)
    print("\n\nPhase A done. Read the output above before continuing.")
    print("If the teacher transliterates English, STOP and change the teacher.\n")
    # Comment out the next line for an audit-only run.
    label_and_cache(rows)
