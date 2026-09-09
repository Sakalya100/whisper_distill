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
