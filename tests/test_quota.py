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
