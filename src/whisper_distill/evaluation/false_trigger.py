"""False-trigger rate on silence and pure noise.

Whisper hallucinates during silence, and distillation tends to make it worse. For a
dictation app that leaves the mic open, this is the failure users will actually report --
notes filling with "Thanks for watching!" while nobody is speaking. It is invisible to WER,
because WER is only ever computed on utterances that contain speech.

So it gets measured explicitly, and a VAD goes in front of the model in the app.

The probe set is synthetic on purpose: it needs no recording session, it is exactly
reproducible across model versions, and the correct output is unambiguous (empty).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from whisper_distill.config import SAMPLE_RATE

#: Anything shorter than this is not a plausible dictation hallucination; the model
#: emitting a stray token is noise, not a false trigger a user would see.
MIN_HALLUCINATION_CHARS = 3


@dataclass
class FalseTriggerResult:
    by_condition: dict[str, tuple[int, int]]
    examples: dict[str, list[str]]

    @property
    def overall_rate(self) -> float:
        trig = sum(t for t, _ in self.by_condition.values())
        tot = sum(n for _, n in self.by_condition.values())
        return trig / tot if tot else 0.0

    def render(self) -> str:
        lines = [f"false-trigger rate (overall): {self.overall_rate * 100:5.1f}%"]
        for cond in sorted(self.by_condition):
            t, n = self.by_condition[cond]
            lines.append(f"  {cond:<16} {t:>3}/{n:<3} {t / n * 100:5.1f}%")
        for cond, ex in sorted(self.examples.items()):
            if ex:
                lines.append(f"\n  {cond} hallucinations:")
                lines += [f"    {e[:80]!r}" for e in ex[:5]]
        return "\n".join(lines)


def probe_set(
    *,
    seconds: float = 5.0,
    n_per_condition: int = 20,
    seed: int = 0,
) -> dict[str, list[np.ndarray]]:
    """Synthetic non-speech audio across the conditions a phone mic actually sees.

    `digital_silence` is the pathological case (exact zeros, which never occurs in real
    recordings but is what a muted mic produces); the others approximate room tone, a fan,
    and mains hum.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE

    def brown(scale: float) -> np.ndarray:
        """Low-frequency-weighted noise -- closer to fan and traffic rumble than white."""
        w = rng.standard_normal(n)
        b = np.cumsum(w)
        b -= b.mean()
        peak = np.abs(b).max()
        return (b / peak * scale).astype(np.float32) if peak else b.astype(np.float32)

    conditions: dict[str, list[np.ndarray]] = {
        "digital_silence": [np.zeros(n, dtype=np.float32) for _ in range(n_per_condition)],
        "room_tone": [
            (rng.standard_normal(n) * 0.001).astype(np.float32)
            for _ in range(n_per_condition)
        ],
        "fan_noise": [brown(0.05) for _ in range(n_per_condition)],
        "mains_hum": [
            (0.02 * np.sin(2 * np.pi * 50 * t)
             + rng.standard_normal(n) * 0.002).astype(np.float32)
            for _ in range(n_per_condition)
        ],
    }
    return conditions


def measure(
    transcribe: Callable[[np.ndarray], str],
    conditions: dict[str, Sequence[np.ndarray]] | None = None,
    *,
    min_chars: int = MIN_HALLUCINATION_CHARS,
) -> FalseTriggerResult:
    """Run `transcribe` over the probe set and count non-empty outputs.

    `transcribe` takes a float32 mono array at 16 kHz and returns the model's text. Any
    output with at least `min_chars` non-whitespace characters counts as a false trigger --
    the correct answer on every clip here is nothing at all.
    """
    conditions = conditions or probe_set()
    by_condition: dict[str, tuple[int, int]] = {}
    examples: dict[str, list[str]] = {}

    for cond, clips in conditions.items():
        triggered = 0
        seen: list[str] = []
        for clip in clips:
            text = (transcribe(clip) or "").strip()
            if len(text) >= min_chars:
                triggered += 1
                seen.append(text)
        by_condition[cond] = (triggered, len(clips))
        examples[cond] = seen

    return FalseTriggerResult(by_condition, examples)
