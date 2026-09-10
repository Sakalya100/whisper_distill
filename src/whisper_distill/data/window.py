"""Cost the candidate input windows of decision 0004 against a real duration mix.

Whisper pays the **full window cost on every step regardless of how much of it is
speech**, so window length is not a tuning knob -- it sets the price of every training
step and every on-device inference, and it is frozen into the mel cache the moment the
shards are written. Decision 0004 lists three candidates (10 s, 6 s, bucketed) and defers
the choice until IndicVoices is profiled.

This module is the arithmetic that closes it. Two things it is careful about:

**Clip count, not hours, is what padding costs are paid in.** A step costs one window
regardless of the clip's length, so a source contributes steps in proportion to
``hours / mean_duration``, not to hours. Vaani at a 2.60 s median contributes far more
steps per hour than a long-form source does, which drags the mix toward its short clips.
Weighting by hours understates that.

**Truncation is reported separately from padding, never netted against it.** Shrinking the
window always improves padding and always worsens truncation; a single blended score would
let one hide the other. Decision 0004 already rules that "the window must serve the task,
not the corpus", so ``speech_lost_fraction`` is the constraint and padding is the thing
being minimised subject to it.

Pure Python, unit-tested, no torch. Run it on the JSON that
``kaggle/01a_corpus_distribution_cpu.py`` writes:

    python -m whisper_distill.data.window corpus_profile_*.json
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

#: Encoder compute is treated as linear in window length, matching the plan's claim that a
#: 10 s window is "~3x less encoder compute" than Whisper's native 30 s. Self-attention is
#: quadratic in sequence length, so this UNDERSTATES the saving from a shorter window --
#: the bias is toward the status quo, which is the safe direction for a decision like this.
BASELINE_WINDOW_S = 10.0

def quantile_samples(quantiles: Mapping[float, float], n: int = 2000) -> list[float]:
    """Reconstruct pseudo-samples from a sparse quantile table.

    01a stores six percentiles, not the raw durations, so an exact expectation is not
    recoverable. Piecewise-linear interpolation of the quantile function is the honest
    reconstruction: it is exact at the measured points and monotone between them.

    The tail is where this hurts: a straight line from p90 to the max is far too fat,
    because real duration tails are convex. On Vaani it predicts 5.6% of clips over 10 s
    against 2.0% measured. That is why ``from_profile_json`` feeds in the profile's own
    ``clips_over_10s_pct`` as an extra anchor -- pass the measured tail if you have it.
    """
    if not quantiles:
        raise ValueError("no quantiles given")
    pts = sorted((float(p), float(v)) for p, v in quantiles.items())
    if pts[0][0] > 0.0:
        # Extend to p0 along the first measured segment's slope, floored at zero.
        (p0, v0), (p1, v1) = pts[0], (pts[1] if len(pts) > 1 else pts[0])
        slope = (v1 - v0) / (p1 - p0) if p1 > p0 else 0.0
        pts.insert(0, (0.0, max(0.0, v0 - slope * p0)))

    out = []
    for i in range(n):
        p = (i + 0.5) / n
        for (p0, v0), (p1, v1) in pairwise(pts):
            if p <= p1:
                span = p1 - p0
                out.append(v0 if span <= 0 else v0 + (v1 - v0) * (p - p0) / span)
                break
        else:
            out.append(pts[-1][1])
    return out


@dataclass(frozen=True)
class SourceProfile:
    """One corpus's measured duration distribution plus its weight in the data plan."""

    name: str
    target_hours: float
    durations: tuple[float, ...]

    @classmethod
    def from_profile_json(cls, path: Path, target_hours: float) -> SourceProfile:
        """Load a JSON written by kaggle/01a_corpus_distribution_cpu.py."""
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        d = blob["duration_s"]
        table = {
            0.10: d["p10"], 0.25: d["p25"], 0.50: d["median"],
            0.75: d["p75"], 0.90: d["p90"], 1.00: d["max"],
        }
        # The measured over-10 s share is a seventh quantile in disguise, and it lands in
        # exactly the stretch the p90->max line gets most wrong. Without it the tail is
        # roughly 3x too fat on Vaani, which would overstate every truncation number here.
        # The endpoints are admitted rather than special-cased: 0% resolves to q(1.0),
        # where the measured max already sits and setdefault leaves it alone, and 100%
        # resolves to q(0.0), which is a real lower bound worth having.
        over_10 = blob.get("clips_over_10s_pct")
        if over_10 is not None and 0.0 <= over_10 <= 100.0:
            table.setdefault(1.0 - over_10 / 100.0, BASELINE_WINDOW_S)
        return cls(
            name=blob.get("dataset", Path(path).stem),
            target_hours=target_hours,
            durations=tuple(quantile_samples(table)),
        )

    @property
    def mean_duration_s(self) -> float:
        return sum(self.durations) / len(self.durations)

    @property
    def clip_weight(self) -> float:
        """Relative number of training steps this source contributes.

        Steps, not hours: ``hours / mean_duration`` is the clip count, and one clip is one
        step at one full window's cost however short the clip is.
        """
        return self.target_hours * 3600.0 / self.mean_duration_s


@dataclass(frozen=True)
class WindowCost:
    """What one candidate window costs against a given duration mix."""

    window_s: float
    padding_fraction: float
    clips_truncated_fraction: float
    speech_lost_fraction: float
    relative_encoder_compute: float

    @property
    def speech_per_unit_compute(self) -> float:
        """Retained speech per unit of encoder compute, relative to the baseline window.

        The efficiency number. Higher is better, but it is meaningless on its own --
        a 1 s window scores superbly and truncates almost everything.
        """
        if self.relative_encoder_compute <= 0:
            return 0.0
        return (1.0 - self.speech_lost_fraction) / self.relative_encoder_compute


def window_cost(
    durations: Sequence[float],
    window_s: float,
    weights: Sequence[float] | None = None,
    baseline_s: float = BASELINE_WINDOW_S,
) -> WindowCost:
    """Cost one window against a weighted set of clip durations."""
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    if not durations:
        raise ValueError("no durations given")
    w = list(weights) if weights is not None else [1.0] * len(durations)
    if len(w) != len(durations):
        raise ValueError("weights and durations differ in length")

    total_w = sum(w)
    kept = sum(wi * min(d, window_s) for d, wi in zip(durations, w))
    speech = sum(wi * d for d, wi in zip(durations, w))
    over = sum(wi for d, wi in zip(durations, w) if d > window_s)

    return WindowCost(
        window_s=window_s,
        padding_fraction=1.0 - kept / (total_w * window_s),
        clips_truncated_fraction=over / total_w,
        speech_lost_fraction=(speech - kept) / speech if speech else 0.0,
        relative_encoder_compute=window_s / baseline_s,
    )


def mixed_cost(
    sources: Sequence[SourceProfile],
    window_s: float,
    baseline_s: float = BASELINE_WINDOW_S,
) -> WindowCost:
    """Cost one window against the source mix, weighted by contributed training steps."""
    if not sources:
        raise ValueError("no sources given")
    durations: list[float] = []
    weights: list[float] = []
    for s in sources:
        per_clip = s.clip_weight / len(s.durations)
        durations.extend(s.durations)
        weights.extend([per_clip] * len(s.durations))
    return window_cost(durations, window_s, weights, baseline_s=baseline_s)


def render_comparison(
    sources: Sequence[SourceProfile],
    candidates: Sequence[float] = (4.0, 6.0, 8.0, 10.0),
    baseline_s: float = BASELINE_WINDOW_S,
) -> str:
    """A table for pasting into decision 0004, plus the step-share breakdown behind it."""
    lines = ["sources (weighted by contributed training steps, not hours)"]
    total = sum(s.clip_weight for s in sources)
    for s in sources:
        lines.append(
            f"  {s.name:<52} {s.target_hours:>6.0f} h  mean {s.mean_duration_s:5.2f} s"
            f"  {s.clip_weight / total * 100:5.1f}% of steps"
        )

    lines += [
        "",
        (f"{'window':>7}  {'padding':>8}  {'clips cut':>10}  {'speech lost':>12}"
         f"  {'compute':>8}  {'speech/compute':>15}"),
        "  " + "-" * 68,
    ]
    for w in candidates:
        c = mixed_cost(sources, w, baseline_s=baseline_s)
        mark = "  <- baseline" if abs(w - baseline_s) < 1e-9 else ""
        lines.append(
            f"{c.window_s:>6.1f}s  {c.padding_fraction * 100:>7.1f}%"
            f"  {c.clips_truncated_fraction * 100:>9.1f}%"
            f"  {c.speech_lost_fraction * 100:>11.2f}%"
            f"  {c.relative_encoder_compute:>7.2f}x"
            f"  {c.speech_per_unit_compute:>14.2f}{mark}"
        )
    lines += [
        "",
        "padding     share of each window that is not speech -- paid on every step",
        "clips cut   share of clips that hit the ceiling and lose their tail",
        "speech lost share of total speech seconds discarded by truncation",
        "compute     encoder cost relative to the 10 s baseline (linear; see module docs)",
        "",
        "Decision 0004: the window must serve the task, not the corpus. Dictation runs",
        "3-10 s, so read 'speech lost' as the constraint and minimise padding under it.",
    ]
    return "\n".join(lines)


def _main(argv: Sequence[str]) -> int:
    from whisper_distill.config import DEFAULT

    if not argv:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2

    by_hf_id = {s.hf_id: s for s in DEFAULT.data.sources}
    sources, seen = [], set()
    for arg in argv:
        blob = json.loads(Path(arg).read_text(encoding="utf-8"))
        hf_id = str(blob.get("dataset", "")).split(":")[0]
        declared = by_hf_id.get(hf_id)
        if declared is None:
            print(f"{arg}: {hf_id!r} is not in DataConfig.sources -- weighting it 0 h "
                  "would drop it, so give it a target in config.py first", file=sys.stderr)
            return 1
        seen.add(hf_id)
        sources.append(SourceProfile.from_profile_json(Path(arg), declared.target_hours))

    missing = [s.hf_id for s in DEFAULT.data.sources if s.hf_id not in seen]
    print(render_comparison(sources))
    if missing or DEFAULT.data.scraped_hinglish_hours:
        print("\nPROVISIONAL -- not every planned source is measured yet:")
        for m in missing:
            print(f"  {m}: declared in DataConfig.sources, no profile JSON given")
        if DEFAULT.data.scraped_hinglish_hours:
            print(f"  scraped Hinglish ({DEFAULT.data.scraped_hinglish_hours:.0f} h): "
                  "not yet collected, distribution unknown")
        print("Do not close decision 0004 on a partial mix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
