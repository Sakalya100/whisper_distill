"""Version-agnostic audio decoding for Hugging Face dataset rows.

`datasets` 4.0 changed the Audio feature: `row["audio"]` now returns a torchcodec
`AudioDecoder` rather than a `{"array", "sampling_rate"}` dict. Legacy dict indexing is
reportedly still supported, but torchcodec also needs a matching torch build and a system
FFmpeg, and there are open issues where that combination fails outright. Kaggle's
preinstalled `datasets` version is not something we control or can pin cheaply.

So this module accepts every shape the field can take and says which path it used:

  1. ``{"array": np.ndarray, "sampling_rate": int}``   -- datasets 3.x
  2. ``AudioDecoder`` with ``.get_all_samples()``      -- datasets 4.x
  3. ``{"path": str, "bytes": bytes | None}``          -- ``Audio(decode=False)``

Path 3 is the one to prefer for a long unattended run: it bypasses torchcodec entirely and
decodes with soundfile, which is stable across versions. See `stream_audio_rows`.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np

from whisper_distill.config import SAMPLE_RATE

log = logging.getLogger(__name__)


def to_mono(wav: np.ndarray) -> np.ndarray:
    """Average any channel layout down to mono float32.

    Handles both (n, channels) from soundfile and (channels, n) from torchcodec by
    treating the shorter axis as channels -- audio is always far longer than it is wide.
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 1:
        return wav
    if wav.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {wav.shape}")
    axis = 0 if wav.shape[0] < wav.shape[1] else 1
    return wav.mean(axis=axis).astype(np.float32)


def resample(wav: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    """Resample to `sr_out`, preferring librosa and falling back to linear interpolation.

    The fallback is deliberately available: librosa pulls in numba, which has been a
    frequent source of version conflicts on hosted images. Linear interpolation is worse
    than a proper polyphase filter but it is not catastrophic for a mel front end, and a
    run that completes beats a run that dies on an import.
    """
    if sr_in == sr_out:
        return np.asarray(wav, dtype=np.float32)
    try:
        import librosa

        return librosa.resample(
            np.asarray(wav, dtype=np.float32), orig_sr=sr_in, target_sr=sr_out
        )
    except ImportError:
        log.warning("librosa unavailable; falling back to linear resampling")
        n_out = int(round(len(wav) * sr_out / sr_in))
        return np.interp(
            np.linspace(0.0, len(wav) - 1, n_out),
            np.arange(len(wav)),
            np.asarray(wav, dtype=np.float32),
        ).astype(np.float32)


def decode_audio_field(field: Any, *, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, str]:
    """Decode one dataset audio field to mono float32 at `target_sr`.

    Returns ``(wav, path_taken)``. The second value names which branch handled it, so a
    run's log records what the environment actually did rather than what we assumed.
    """
    # --- 1. datasets 3.x dict, and 4.x legacy indexing
    if isinstance(field, dict) and "array" in field:
        sr = int(field.get("sampling_rate") or target_sr)
        return resample(to_mono(field["array"]), sr, target_sr), "dict_array"

    # --- 3. Audio(decode=False): raw bytes or a path
    if isinstance(field, dict) and ("bytes" in field or "path" in field):
        import soundfile as sf

        raw = field.get("bytes")
        src = io.BytesIO(raw) if raw else field.get("path")
        if src is None:
            raise ValueError("audio field has neither bytes nor a usable path")
        wav, sr = sf.read(src, dtype="float32", always_2d=False)
        return resample(to_mono(wav), int(sr), target_sr), "soundfile"

    # --- 2. datasets 4.x AudioDecoder
    if hasattr(field, "get_all_samples"):
        samples = field.get_all_samples()
        data = samples.data
        if hasattr(data, "numpy"):  # torch tensor
            data = data.numpy()
        return (
            resample(to_mono(data), int(samples.sample_rate), target_sr),
            "audio_decoder",
        )

    # A bare path string, which some loaders still hand back.
    if isinstance(field, str):
        import soundfile as sf

        wav, sr = sf.read(field, dtype="float32", always_2d=False)
        return resample(to_mono(wav), int(sr), target_sr), "soundfile_path"

    raise TypeError(
        f"unrecognised audio field of type {type(field).__name__}. "
        "Add a branch to decode_audio_field rather than guessing at the call site."
    )


#: Column names corpora use for the reference text, in the order we prefer them.
#: Vaani uses `transcript`; IndicVoices publishes no schema on its card, so the key has to
#: be discovered rather than assumed. Ordered so a genuine transcript beats a normalised
#: or verbatim variant when a corpus ships several.
TRANSCRIPT_KEY_CANDIDATES = (
    "transcript",
    "text",
    "sentence",
    "transcription",
    "normalized_text",
    "verbatim",
    "raw_text",
)


def find_transcript_key(
    row: dict, candidates: tuple[str, ...] = TRANSCRIPT_KEY_CANDIDATES
) -> str | None:
    """Discover which column holds the reference text.

    Prefers a candidate that is actually populated over one that merely exists -- a corpus
    can carry an empty `text` alongside a filled `transcript`, and picking the empty one
    silently disables the WER filter downstream.
    """
    present = [c for c in candidates if c in row]
    for c in present:
        value = row.get(c)
        if isinstance(value, str) and value.strip():
            return c
    return present[0] if present else None


def find_audio_key(row: dict) -> str | None:
    """Discover which column holds the audio, by shape rather than by name."""
    for name in ("audio", "audio_filepath", "wav", "speech"):
        if name in row:
            return name
    for name, value in row.items():
        if hasattr(value, "get_all_samples"):
            return name
        if isinstance(value, dict) and ({"array", "bytes", "path"} & set(value)):
            return name
    return None


def describe_row(row: dict) -> str:
    """Human-readable summary of an unknown corpus row. Use before assuming anything."""
    audio_key = find_audio_key(row)
    text_key = find_transcript_key(row)
    lines = [
        f"columns       : {sorted(row)}",
        f"audio column  : {audio_key!r}",
        f"text column   : {text_key!r}",
    ]
    if audio_key:
        field = row[audio_key]
        shape = type(field).__name__
        if isinstance(field, dict):
            shape += f" keys={sorted(field)}"
        lines.append(f"audio field   : {shape}")
    if text_key:
        lines.append(f"text sample   : {str(row.get(text_key))[:100]!r}")
    other = {
        k: (str(v)[:40] if not isinstance(v, (dict, list)) else type(v).__name__)
        for k, v in row.items()
        if k not in {audio_key, text_key}
    }
    lines.append(f"other columns : {other}")
    return "\n".join(lines)


def probe_schema(row: dict, *, require: tuple[str, ...] = ("audio", "transcript")) -> str:
    """Assert the columns we depend on exist, and describe the audio field's shape.

    Called on the FIRST streamed row so a schema mismatch costs ten seconds instead of an
    hour. Vaani's transcribed part exposes audio, language, gender, state, district,
    transcript and referenceImage -- but a config or a library version can change that,
    and the WER filter in step 2 is silently useless if `transcript` arrives empty.
    """
    missing = [c for c in require if c not in row]
    if missing:
        raise KeyError(
            f"dataset row is missing {missing}; available columns are "
            f"{sorted(row)}. Fix the column names before streaming a whole corpus."
        )
    field = row["audio"]
    shape = type(field).__name__
    if isinstance(field, dict):
        shape += f" keys={sorted(field)}"
    return shape
