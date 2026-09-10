"""Step 1a -- measure the corpus before committing to a window size.

    Accelerator: NONE (CPU)     Internet: ON     Quota: FREE     Runtime: ~3-5 min

WHY THIS EXISTS
  The schema probe's first row decoded to 2.0 seconds, carrying one complete short
  utterance ("बहुत ही सुन्दर टमाटर है ।"). If that is typical rather than an outlier, two
  design assumptions in the plan are wrong:

    1. The 10 s student window would be ~80% padding. That wastes encoder compute on every
       step and, worse, the model never sees a long utterance during training -- while real
       dictation ("remind me to call mom at 4 tomorrow, and add milk to the list") is
       3-10 s. Optimising the window for 2 s clips would build a model that fails on the
       actual task.
    2. VAD segmentation would be close to pointless -- there is nothing to segment. That is
       a whole CPU stage doing nothing, and merge_to_window's max_gap_s tuning would be
       noise.

  Also: that transcript is pure Devanagari with no Latin script at all. If Vaani Hindi is
  not meaningfully code-mixed, it is fine as acoustic coverage but it is NOT the source of
  the Hinglish behaviour this project is about -- and the data plan's 120/50/30 hour split
  needs rebalancing toward IndicVoices and scraped audio.

  One sample decides nothing. This measures the distribution.

WHERE THIS STANDS
  Vaani Hindi is measured (400 rows: median 2.60 s, 2.0% over 10 s, 68.2% Latin-script --
  see docs/research/2026-09-09-corpus-schema-probe.md). Assumption 1 held, assumption 2
  held, and the code-mixing worry did not. What is still open is the window length, and
  that CANNOT be decided from Vaani: it is pre-segmented short utterances, so it carries
  acoustic and speaker coverage but not utterance-length coverage.

  So the remaining job is the IndicVoices leg. Flip SOURCE below, commit, then feed both
  JSONs to the cost model, which weights by contributed training steps rather than hours:

      python -m whisper_distill.data.window corpus_profile_*.json

  That closes docs/decisions/0004-input-window-length.md.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

N_ROWS = 400           # ~3-5 min. Enough for stable deciles, cheap enough to redo.

# One line switches corpus. These mirror DataConfig.sources in config.py -- keep them in
# step. The window must come from the COMBINED distribution weighted by the final source
# mix, so run this once per source and keep both JSONs.
SOURCES = {
    "vaani": ("ARTPARK-IISc/Vaani-transcription-part", "Hindi", "train"),
    "indicvoices": ("ai4bharat/IndicVoices", "hindi", "train"),
}
SOURCE = "indicvoices"          # <-- vaani is already measured; this is the open leg

DATASET, CONFIG, SPLIT = SOURCES[SOURCE]
TRANSCRIPT_KEY = None   # None = discover it. IndicVoices publishes no schema on its card.

OUT = Path(f"/kaggle/working/corpus_profile_{DATASET.split('/')[-1]}_{CONFIG}.json")
REPO_SRC = "/kaggle/working/whisper_distill/src"

sys.path.insert(0, REPO_SRC)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def main() -> None:
    import re

    from datasets import load_dataset
    from kaggle_secrets import UserSecretsClient

    from whisper_distill.config import DEFAULT
    from whisper_distill.data.audio_io import (
        decode_audio_field,
        describe_row,
        find_audio_key,
        find_transcript_key,
    )
    from whisper_distill.data.streaming import take
    from whisper_distill.evaluation.metrics import code_mix_bucket, code_mix_density

    audio_cfg = DEFAULT.audio
    token = UserSecretsClient().get_secret("HF_TOKEN")
    ds = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True, token=token)

    latin = re.compile(r"[A-Za-z]")
    devanagari = re.compile(r"[ऀ-ॿ]")

    durations: list[float] = []
    words: list[int] = []
    densities: list[float] = []
    buckets: Counter[str] = Counter()
    n_latin = n_empty = n_danda = n_comma = 0
    decode_paths: Counter[str] = Counter()
    start = time.time()

    # Discover the schema before profiling. IndicVoices ships no column list, so
    # hardcoding a key here would fail 400 rows in with nothing to show for it.
    first = next(iter(take(ds, 1)))
    print(describe_row(first) + "\n")
    audio_key = find_audio_key(first) or "audio"
    text_key = TRANSCRIPT_KEY or find_transcript_key(first)
    if text_key is None:
        raise SystemExit(
            f"no reference-text column found in {sorted(first)}. Set TRANSCRIPT_KEY "
            "explicitly at the top of this script."
        )
    print(f"using audio={audio_key!r} text={text_key!r}\n")
    del first

    print(f"profiling {N_ROWS} rows of {DATASET}:{CONFIG}:{SPLIT}\n", flush=True)
    # take() rather than `for row in ds` + break: abandoning the stream mid-download
    # leaves hub retry threads and FFmpeg workers running into interpreter shutdown, which
    # surfaces as a fatal PyGILState_Release error. Harmless after the work is flushed, but
    # on Kaggle a fatal error can mark the commit failed and block dataset creation.
    for i, row in enumerate(take(ds, N_ROWS)):
        try:
            wav, how = decode_audio_field(row[audio_key], target_sr=audio_cfg.sample_rate)
        except Exception as e:  # noqa: BLE001
            print(f"  row {i}: decode failed ({type(e).__name__}); skipping")
            continue
        decode_paths[how] += 1
        durations.append(len(wav) / audio_cfg.sample_rate)

        t = (row.get(text_key) or "").strip()
        if not t:
            n_empty += 1
            continue
        words.append(len(t.split()))
        densities.append(code_mix_density(t))
        buckets[code_mix_bucket(t)] += 1
        n_latin += bool(latin.search(t))
        n_danda += "।" in t
        n_comma += ("," in t or "," in t)

        if i and i % 50 == 0:
            print(f"  {i:>4}/{N_ROWS}  {(time.time() - start) / 60:.1f} min", flush=True)

    n = len(durations)
    if not n:
        raise SystemExit("no rows decoded -- check the token and gating")

    total_h = sum(durations) / 3600
    print("\n" + "=" * 68)
    print(f"AUDIO  ({n} rows, {total_h:.3f} h, {(time.time() - start) / 60:.1f} min)")
    print("=" * 68)
    for label, q in (("p10", .10), ("p25", .25), ("median", .50),
                     ("p75", .75), ("p90", .90), ("max", 1.0)):
        print(f"  {label:<7} {pct(durations, q):6.2f} s")
    print(f"  {'mean':<7} {sum(durations) / n:6.2f} s")
    over_10 = sum(d > 10 for d in durations)
    print(f"\n  clips > 10 s : {over_10} / {n}  ({over_10 / n * 100:.1f}%)")
    print(f"  decode paths : {dict(decode_paths)}")

    print("\n" + "=" * 68)
    print("TRANSCRIPTS")
    print("=" * 68)
    print(f"  empty              : {n_empty} / {N_ROWS}")
    print(f"  median words       : {pct([float(w) for w in words], .50):.0f}")
    print(f"  contains Latin     : {n_latin} / {len(words)}  "
          f"({n_latin / max(len(words), 1) * 100:.1f}%)")
    print(f"  contains danda ।   : {n_danda} / {len(words)}")
    print(f"  contains comma     : {n_comma} / {len(words)}")
    print(f"  mean code-mix dens : {sum(densities) / max(len(densities), 1):.3f}")
    print(f"  buckets            : {dict(buckets)}")

    # ------------------------------------------------------------------ what it means
    median = pct(durations, .50)
    print("\n" + "=" * 68)
    print("READ-OUT")
    print("=" * 68)

    if median < 3.0:
        waste = (1 - median / 10.0) * 100
        print(
            f"  Median clip is {median:.1f} s, so a 10 s window is ~{waste:.0f}% padding.\n"
            "  ACTION: do not tune the window for this corpus alone -- it is pre-segmented\n"
            "  short utterances, which is acoustic and speaker coverage, not\n"
            "  utterance-length coverage. Real dictation runs 3-10 s. Profile the other\n"
            "  sources and settle the window on the weighted mix.\n"
            "  Also: VAD has almost nothing to segment here -- keep it only to trim\n"
            "  leading/trailing silence, and skip merge_to_window for this source."
        )
    elif median > 8.0:
        print(
            f"  Median clip is {median:.1f} s, close to the 10 s ceiling. Segmentation is\n"
            "  doing real work and the window is well matched. Proceed as planned."
        )
    else:
        print(
            f"  Median clip is {median:.1f} s -- a good match for a 10 s window with room\n"
            "  for longer utterances. Proceed as planned."
        )

    latin_pct = n_latin / max(len(words), 1) * 100
    if latin_pct < 5.0:
        print(
            f"\n  Only {latin_pct:.1f}% of transcripts contain any Latin script -- this\n"
            "  source is essentially monolingual Devanagari.\n"
            "  ACTION: it cannot teach code-switching. Keep it for acoustic robustness and\n"
            "  speaker/accent diversity, but the Hinglish behaviour has to come from\n"
            "  IndicVoices and the scraped corpus. Revisit the data plan's hour split, and\n"
            "  note that Gate 1's code-switch audit needs genuinely code-mixed audio --\n"
            "  which means it CANNOT be run on this source alone."
        )
    else:
        print(f"\n  {latin_pct:.1f}% of transcripts contain Latin script -- usable code-mixing.")

    OUT.write_text(json.dumps({
        "dataset": f"{DATASET}:{CONFIG}:{SPLIT}",
        "n_rows": n,
        "hours_sampled": total_h,
        "duration_s": {
            "p10": pct(durations, .10), "p25": pct(durations, .25),
            "median": median, "p75": pct(durations, .75),
            "p90": pct(durations, .90), "max": pct(durations, 1.0),
            "mean": sum(durations) / n,
        },
        "clips_over_10s_pct": over_10 / n * 100,
        "transcripts": {
            "empty": n_empty,
            "median_words": pct([float(w) for w in words], .50),
            "latin_pct": latin_pct,
            "danda": n_danda,
            "comma": n_comma,
            "mean_code_mix_density": sum(densities) / max(len(densities), 1),
            "buckets": dict(buckets),
        },
        "decode_paths": dict(decode_paths),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print("\nDownload this JSON, put it beside the other sources' profiles, and run\n"
          "  python -m whisper_distill.data.window corpus_profile_*.json\n"
          "to cost the windows of decision 0004 against the weighted mix.")


if __name__ == "__main__":
    main()
