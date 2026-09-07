"""Decode a Kaggle GPU-meter reading into a billing verdict.

Gate 0 of the plan. The 30 h/week pool is shared across P100 and T4x2, and nothing in
Kaggle's docs states whether the meter charges session wall-clock or GPU-hours. It moves
the schedule ~2x and decides whether the three-ablation block is affordable.

With `n_sessions` concurrent T4x2 commits each running `minutes` of wall-clock:

    wall-clock billing  ->  delta ~= n_sessions * minutes
    per-GPU billing     ->  delta ~= n_sessions * minutes * 2

Pure Python, unit-tested. Run `decode_quota_probe` on the real numbers once the sessions
finish and write the verdict into docs/01-kaggle-execution-plan.md.
"""

from __future__ import annotations

from dataclasses import dataclass

GPUS_PER_T4X2 = 2

#: Committed runs pay container startup, dependency install and teardown on top of the
#: cell's own runtime. Compare against the notebook's reported duration, not your sleep().
STARTUP_OVERHEAD_TOLERANCE = 0.35


@dataclass(frozen=True)
class QuotaVerdict:
    billing: str  # "wall_clock" | "per_gpu" | "ambiguous"
    observed_minutes: float
    expected_wall_clock: float
    expected_per_gpu: float
    total_quota_hours_needed: float
    weeks_at_30h: float
    note: str

    def render(self) -> str:
        return "\n".join(
            [
                f"observed meter delta : {self.observed_minutes:.1f} min",
                f"  if wall-clock      : {self.expected_wall_clock:.1f} min",
                f"  if per-GPU         : {self.expected_per_gpu:.1f} min",
                "",
                f"VERDICT: {self.billing}",
                f"  budget  : {self.total_quota_hours_needed:.0f} quota-hours",
                f"  schedule: {self.weeks_at_30h:.1f} weeks at 30 h/week",
                "",
                self.note,
            ]
        )


def decode_quota_probe(
    *,
    meter_delta_minutes: float,
    n_concurrent_sessions: int,
    minutes_per_session: float,
    gpus_per_session: int = GPUS_PER_T4X2,
    gpu_hours_needed: float = 108.0,
    tolerance: float = STARTUP_OVERHEAD_TOLERANCE,
) -> QuotaVerdict:
    """Classify the meter reading and translate it into a schedule.

    `gpu_hours_needed` is the plan's mid-estimate of compute required, in GPU-hours. Under
    wall-clock billing a dual-GPU session converts 1 quota-hour into 2 GPU-hours, so the
    quota cost halves.
    """
    if n_concurrent_sessions < 1 or minutes_per_session <= 0:
        raise ValueError("need at least one session with positive runtime")

    wall = n_concurrent_sessions * minutes_per_session
    per_gpu = wall * gpus_per_session

    # Classify by which prediction is closer, but refuse to call it when the reading lands
    # near the midpoint -- a proportional tolerance around each target would quietly favour
    # the larger one, which is exactly the wrong bias for a budget decision.
    midpoint = (wall + per_gpu) / 2
    ambiguous = abs(meter_delta_minutes - midpoint) <= midpoint * tolerance * 0.5
    off_scale = (
        meter_delta_minutes < wall * (1 - tolerance)
        or meter_delta_minutes > per_gpu * (1 + tolerance)
    )

    if not ambiguous and not off_scale and meter_delta_minutes < midpoint:
        billing = "wall_clock"
        quota_hours = gpu_hours_needed / gpus_per_session
        note = (
            "Session wall-clock is what gets charged. Run everything as DDP inside ONE "
            "T4x2 session -- two concurrent single-GPU sessions bill twice for the same "
            "work. Concurrent T4x2 sessions are still the fastest way to burn through the "
            "ablation block if you have the quota to spend."
        )
    elif not ambiguous and not off_scale:
        billing = "per_gpu"
        quota_hours = gpu_hours_needed
        note = (
            "Each GPU is metered. T4x2 buys only VRAM headroom, not cheaper compute. Cut "
            "the ablation block from three runs to one, or prefer P100 for single-GPU "
            "stages and keep T4x2 for the runs that genuinely need 32 GB."
        )
    else:
        billing = "ambiguous"
        quota_hours = gpu_hours_needed
        note = (
            "Inconclusive: the reading sits near the midpoint of the two predictions, or "
            "outside both. Most likely the session runtimes were unequal, or startup "
            "overhead was large relative to a short run. Re-run with a SINGLE 30-minute "
            "T4x2 commit and nothing else active -- one session removes the concurrency "
            "variable. Budget assumes the pessimistic case until then."
        )

    return QuotaVerdict(
        billing=billing,
        observed_minutes=meter_delta_minutes,
        expected_wall_clock=wall,
        expected_per_gpu=per_gpu,
        total_quota_hours_needed=quota_hours,
        weeks_at_30h=quota_hours / 30.0,
        note=note,
    )
