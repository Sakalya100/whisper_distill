"""Gate 0 decoder. Errs pessimistic when the reading is inconclusive."""

import pytest

from whisper_distill.labeling.quota import decode_quota_probe


def _probe(delta, sessions=2, minutes=15):
    return decode_quota_probe(
        meter_delta_minutes=delta,
        n_concurrent_sessions=sessions,
        minutes_per_session=minutes,
    )


def test_wall_clock_reading_halves_the_quota_cost():
    """Two 15-min T4x2 commits: wall-clock predicts +30, per-GPU predicts +60."""
    v = _probe(30)
    assert v.billing == "wall_clock"
    assert v.total_quota_hours_needed == pytest.approx(54.0)
    assert v.weeks_at_30h == pytest.approx(1.8)


def test_per_gpu_reading_keeps_the_full_cost():
    v = _probe(60)
    assert v.billing == "per_gpu"
    assert v.total_quota_hours_needed == pytest.approx(108.0)


def test_midpoint_is_refused_rather_than_guessed():
    """A budget decision must not be made by rounding toward the nearer target."""
    assert _probe(45).billing == "ambiguous"


def test_ambiguous_assumes_the_pessimistic_budget():
    assert _probe(45).total_quota_hours_needed == _probe(60).total_quota_hours_needed


def test_readings_outside_both_predictions_are_refused():
    assert _probe(3).billing == "ambiguous"
    assert _probe(300).billing == "ambiguous"


def test_startup_overhead_does_not_flip_the_verdict():
    """Committed runs pay container startup on top of the cell's own runtime."""
    assert _probe(33).billing == "wall_clock"
    assert _probe(66).billing == "per_gpu"


def test_single_session_probe_removes_the_concurrency_variable():
    assert _probe(30, sessions=1, minutes=30).billing == "wall_clock"
    assert _probe(60, sessions=1, minutes=30).billing == "per_gpu"


def test_invalid_probe_parameters_are_rejected():
    with pytest.raises(ValueError):
        _probe(30, sessions=0)
    with pytest.raises(ValueError):
        _probe(30, minutes=0)


# ---------------------------------------------------------------- unequal durations
def test_unequal_session_durations_are_summed_not_averaged():
    """Concurrent commits rarely run for equal lengths -- one is started after the other."""
    from whisper_distill.labeling.quota import decode_quota_probe as d

    # 15 + 12 = 27 min of wall clock; x2 GPUs = 54 min under per-GPU billing.
    assert d(meter_delta_minutes=54, session_minutes=[15, 12]).billing == "per_gpu"
    # 27 + 27 = 54 min of wall clock, which per-session billing would charge directly.
    assert d(meter_delta_minutes=54, session_minutes=[27, 27]).billing == "wall_clock"


def test_session_minutes_is_validated():
    from whisper_distill.labeling.quota import decode_quota_probe as d

    with pytest.raises(ValueError):
        d(meter_delta_minutes=30, session_minutes=[])
    with pytest.raises(ValueError):
        d(meter_delta_minutes=30, session_minutes=[15, -1])
    with pytest.raises(ValueError, match="pass session_minutes"):
        d(meter_delta_minutes=30)


# ------------------------------------------------------------------- inverse solve
def test_implied_runtimes_inverts_the_probe():
    """The direction you need when the meter is in hand but the durations are not."""
    from whisper_distill.labeling.quota import implied_runtimes

    r = implied_runtimes(54, n_sessions=2, gpus_per_session=2)
    assert r.if_wall_clock == 54.0   # durations summed to 54 min
    assert r.if_per_gpu == 27.0      # durations summed to 27 min


def test_implied_runtimes_is_identity_on_a_single_gpu():
    """P100 cannot distinguish the hypotheses -- one GPU means both predict the same."""
    from whisper_distill.labeling.quota import implied_runtimes

    r = implied_runtimes(20, n_sessions=1, gpus_per_session=1)
    assert r.if_wall_clock == r.if_per_gpu == 20.0


def test_implied_runtimes_rejects_nonsense():
    from whisper_distill.labeling.quota import implied_runtimes

    with pytest.raises(ValueError):
        implied_runtimes(-1)
    with pytest.raises(ValueError):
        implied_runtimes(30, gpus_per_session=0)
