"""VAD segmentation to <=10 s windows. Runs in a free CPU session.

This is a **prerequisite, not an optimisation**. The fused label+mel pass depends on a
clip's student mel (80x1000) being a prefix of the teacher's 30 s padded mel (80x3000),
and that only holds if the clip is already <=10 s when the teacher sees it.

`merge_to_window` is pure Python and unit-tested; `segment_file` needs silero-vad.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from whisper_distill.config import DEFAULT

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def merge_to_window(
    speech: Sequence[Segment],
    *,
    max_seconds: float = 10.0,
    min_seconds: float = 1.0,
    max_gap_s: float = 0.6,
) -> list[Segment]:
    """Greedily merge adjacent speech regions into clips of at most `max_seconds`.

    Raw VAD output is far too fragmented for training clips -- a single sentence comes back
    as five regions. Merging across gaps up to `max_gap_s` keeps utterances intact, and the
    hard `max_seconds` ceiling is what makes the prefix property hold.

    A region longer than `max_seconds` on its own is split at the boundary rather than
    dropped: it is usually continuous speech, which is exactly what we want.

    Clips shorter than `min_seconds` after merging are dropped -- they are almost always
    breath or a clipped word, and they inflate the clip count without adding audio.
    """
    if max_seconds <= 0 or min_seconds < 0:
        raise ValueError("max_seconds must be > 0 and min_seconds >= 0")

    out: list[Segment] = []
    cur: Segment | None = None

    for seg in sorted(speech, key=lambda s: s.start_s):
        if seg.duration_s <= 0:
            continue
        # Split an over-long region into back-to-back full windows.
        if seg.duration_s > max_seconds:
            if cur is not None:
                out.append(cur)
                cur = None
            t = seg.start_s
            while t < seg.end_s:
                out.append(Segment(t, min(t + max_seconds, seg.end_s)))
                t += max_seconds
            continue

        if cur is None:
            cur = seg
            continue

        gap = seg.start_s - cur.end_s
        if gap <= max_gap_s and (seg.end_s - cur.start_s) <= max_seconds:
            cur = Segment(cur.start_s, seg.end_s)
        else:
            out.append(cur)
            cur = seg

    if cur is not None:
        out.append(cur)
    return [s for s in out if s.duration_s >= min_seconds]


def segment_file(path: str, *, max_seconds: float | None = None) -> list[Segment]:
    """Run silero-vad over one audio file and return merged <=max_seconds clips.

    silero is loaded via torch.hub, which needs network on first call -- so in a Kaggle
    notebook this requires internet enabled (phone verification gates it).
    """
    import torch

    cfg = DEFAULT.data
    max_seconds = max_seconds or cfg.max_clip_seconds

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    get_speech_timestamps, _, read_audio, *_ = utils

    wav = read_audio(path, sampling_rate=DEFAULT.audio.sample_rate)
    stamps = get_speech_timestamps(
        wav, model, sampling_rate=DEFAULT.audio.sample_rate, return_seconds=True
    )
    speech = [Segment(float(s["start"]), float(s["end"])) for s in stamps]
    return merge_to_window(
        speech, max_seconds=max_seconds, min_seconds=cfg.min_clip_seconds
    )
