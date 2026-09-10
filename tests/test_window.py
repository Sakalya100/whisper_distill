"""Decision 0004's padding-cost arithmetic. See src/whisper_distill/data/window.py."""

import json

import pytest

from whisper_distill.data.window import (
    SourceProfile,
    mixed_cost,
    quantile_samples,
    render_comparison,
    segmented_cost,
    window_cost,
)

# Vaani Hindi, 400 rows, from docs/research/2026-09-09-corpus-schema-probe.md.
VAANI = {
    "dataset": "ARTPARK-IISc/Vaani-transcription-part:Hindi:train",
    "duration_s": {"p10": 1.71, "p25": 2.13, "median": 2.60, "p75": 3.79,
                   "p90": 6.35, "max": 14.61, "mean": 3.40},
    "clips_over_10s_pct": 2.0,
}


def test_uniform_clips_at_the_window_length_have_no_padding():
    c = window_cost([10.0] * 50, 10.0)
    assert c.padding_fraction == pytest.approx(0.0)
    assert c.clips_truncated_fraction == pytest.approx(0.0)
    assert c.speech_lost_fraction == pytest.approx(0.0)


def test_half_length_clips_waste_half_the_window():
    c = window_cost([5.0] * 50, 10.0)
    assert c.padding_fraction == pytest.approx(0.5)
    assert c.speech_lost_fraction == pytest.approx(0.0)


def test_truncation_is_reported_separately_from_padding():
    """A 20 s clip in a 10 s window is 0% padding and 50% lost speech, not a wash."""
    c = window_cost([20.0], 10.0)
    assert c.padding_fraction == pytest.approx(0.0)
    assert c.clips_truncated_fraction == pytest.approx(1.0)
    assert c.speech_lost_fraction == pytest.approx(0.5)


def test_compute_scales_linearly_against_the_ten_second_baseline():
    assert window_cost([3.0], 4.0).relative_encoder_compute == pytest.approx(0.4)
    assert window_cost([3.0], 10.0).relative_encoder_compute == pytest.approx(1.0)


def test_shrinking_the_window_always_trades_padding_for_lost_speech():
    """The trade that decision 0004 turns on: neither metric alone can pick a window."""
    s = quantile_samples({0.10: 1.71, 0.50: 2.60, 0.90: 6.35, 1.00: 14.61})
    costs = [window_cost(s, w) for w in (4.0, 6.0, 8.0, 10.0)]
    assert [c.padding_fraction for c in costs] == sorted(c.padding_fraction for c in costs)
    assert [c.speech_lost_fraction for c in costs] == sorted(
        (c.speech_lost_fraction for c in costs), reverse=True)


def test_quantile_reconstruction_is_exact_at_the_measured_points():
    table = {0.10: 1.71, 0.25: 2.13, 0.50: 2.60, 0.75: 3.79, 0.90: 6.35, 1.00: 14.61}
    s = sorted(quantile_samples(table, n=10_000))
    for p, want in table.items():
        got = s[min(len(s) - 1, int(p * (len(s) - 1)))]
        assert got == pytest.approx(want, abs=0.05)


def test_measured_tail_share_anchors_the_reconstruction(tmp_path):
    """Without the over-10 s anchor the p90->max line is ~3x too fat on Vaani."""
    p = tmp_path / "vaani.json"
    p.write_text(json.dumps(VAANI), encoding="utf-8")
    anchored = SourceProfile.from_profile_json(p, 120.0)
    assert mixed_cost([anchored], 10.0).clips_truncated_fraction == pytest.approx(0.02, abs=1e-3)
    assert anchored.mean_duration_s == pytest.approx(3.40, abs=0.15)

    p.write_text(json.dumps({**VAANI, "clips_over_10s_pct": None}), encoding="utf-8")
    unanchored = SourceProfile.from_profile_json(p, 120.0)
    assert mixed_cost([unanchored], 10.0).clips_truncated_fraction > 0.04


def test_a_corpus_with_no_long_clips_is_anchored_not_skipped(tmp_path):
    """0.0 is a measurement, not a missing value -- and it is what 01a writes for a
    corpus whose max is under the window. Skipping it would leave the tail interpolated."""
    p = tmp_path / "short.json"
    short = {**VAANI, "duration_s": {**VAANI["duration_s"], "max": 9.0},
             "clips_over_10s_pct": 0.0}
    p.write_text(json.dumps(short), encoding="utf-8")
    prof = SourceProfile.from_profile_json(p, 120.0)
    assert max(prof.durations) <= 9.0
    assert mixed_cost([prof], 10.0).clips_truncated_fraction == pytest.approx(0.0)


def test_a_corpus_entirely_above_the_window_anchors_at_the_floor(tmp_path):
    """100% over 10 s means q(0) >= 10 s, which is a real lower bound worth keeping."""
    p = tmp_path / "long.json"
    long_ = {**VAANI,
             "duration_s": {"p10": 12.0, "p25": 15.0, "median": 20.0, "p75": 28.0,
                            "p90": 40.0, "max": 61.0, "mean": 24.0},
             "clips_over_10s_pct": 100.0}
    p.write_text(json.dumps(long_), encoding="utf-8")
    prof = SourceProfile.from_profile_json(p, 50.0)
    # Samples are drawn at midpoints, so the floor is approached but never hit.
    assert 10.0 <= min(prof.durations) < 10.1
    assert mixed_cost([prof], 10.0).clips_truncated_fraction == pytest.approx(1.0)


def test_sources_are_weighted_by_steps_not_hours(tmp_path):
    """Equal hours of 2 s and 8 s clips is 4x as many steps from the short source."""
    short = SourceProfile("short", target_hours=10.0, durations=(2.0,) * 100)
    long_ = SourceProfile("long", target_hours=10.0, durations=(8.0,) * 100)
    assert short.clip_weight == pytest.approx(4 * long_.clip_weight)

    # So the mixed mean sits at 3.2 s, near the short source, not at the 5 s hour-average.
    c = mixed_cost([short, long_], 10.0)
    assert c.padding_fraction == pytest.approx(1 - 3.2 / 10.0, abs=1e-6)


def test_a_short_window_scores_well_on_efficiency_while_destroying_the_task():
    """Why speech_per_unit_compute must never be read on its own."""
    s = quantile_samples({0.10: 1.71, 0.50: 2.60, 0.90: 6.35, 1.00: 14.61})
    tiny, full = window_cost(s, 1.0), window_cost(s, 10.0)
    assert tiny.speech_per_unit_compute > full.speech_per_unit_compute
    assert tiny.speech_lost_fraction > 0.5


def test_render_names_every_candidate_and_flags_the_baseline(tmp_path):
    p = tmp_path / "vaani.json"
    p.write_text(json.dumps(VAANI), encoding="utf-8")
    out = render_comparison([SourceProfile.from_profile_json(p, 120.0)])
    assert "% of steps" in out
    assert "<- baseline" in out
    for w in ("4.0s", "6.0s", "8.0s", "10.0s"):
        assert w in out


def test_empty_and_invalid_inputs_are_refused():
    with pytest.raises(ValueError):
        window_cost([], 10.0)
    with pytest.raises(ValueError):
        window_cost([2.0], 0.0)
    with pytest.raises(ValueError):
        window_cost([2.0, 3.0], 10.0, weights=[1.0])
    with pytest.raises(ValueError):
        quantile_samples({})


# --------------------------------------------------------------------------------------
# Segmented model: what merge_to_window actually builds. See segment.py -- an over-long
# region is split into back-to-back windows, not truncated.
# --------------------------------------------------------------------------------------

def test_a_long_clip_is_split_into_windows_not_truncated():
    """25 s at a 10 s window is three clips totalling 25 s of speech, none discarded."""
    sc = segmented_cost([25.0], 10.0)
    assert sc.windows_per_clip == pytest.approx(3.0)
    assert sc.speech_dropped_fraction == pytest.approx(0.0)
    assert sc.padding_fraction == pytest.approx(1 - 25.0 / 30.0)


def test_a_tail_below_min_seconds_is_dropped_rather_than_padded():
    """merge_to_window discards sub-min_seconds remainders as breath or a clipped word."""
    sc = segmented_cost([20.5], 10.0, min_seconds=1.0)
    assert sc.windows_per_clip == pytest.approx(2.0)
    assert sc.speech_dropped_fraction == pytest.approx(0.5 / 20.5)
    assert sc.padding_fraction == pytest.approx(0.0)


def test_a_clip_shorter_than_min_seconds_emits_nothing():
    sc = segmented_cost([0.5], 10.0, min_seconds=1.0)
    assert sc.windows_per_clip == pytest.approx(0.0)
    assert sc.speech_dropped_fraction == pytest.approx(1.0)


def test_segmenting_loses_far_less_speech_than_truncating():
    """The correction that matters: the truncating model overstates loss on long sources."""
    s = quantile_samples({0.10: 0.64, 0.50: 3.52, 0.90: 14.11, 1.00: 28.92})
    assert window_cost(s, 10.0).speech_lost_fraction > 0.15
    assert segmented_cost(s, 10.0).speech_dropped_fraction < 0.02


def test_compute_is_the_whole_corpus_pass_not_the_per_step_cost():
    """Halving the window does not halve the cost -- it emits more windows per clip."""
    s = quantile_samples({0.10: 0.64, 0.50: 3.52, 0.90: 14.11, 1.00: 28.92})
    assert segmented_cost(s, 10.0).relative_encoder_compute == pytest.approx(1.0)
    half = segmented_cost(s, 5.0).relative_encoder_compute
    assert 0.5 < half < 1.0


def test_segmented_padding_still_falls_as_the_window_shrinks():
    s = quantile_samples({0.10: 1.71, 0.50: 2.60, 0.90: 6.35, 1.00: 14.61})
    pads = [segmented_cost(s, w).padding_fraction for w in (4.0, 6.0, 8.0, 10.0)]
    assert pads == sorted(pads)


def test_render_shows_both_cost_models(tmp_path):
    p = tmp_path / "vaani.json"
    p.write_text(json.dumps(VAANI), encoding="utf-8")
    out = render_comparison([SourceProfile.from_profile_json(p, 120.0)])
    assert "AS SEGMENTED" in out
    assert "windows/clip" in out
    assert "speech lost" in out


def test_segmented_refuses_the_same_bad_inputs():
    with pytest.raises(ValueError):
        segmented_cost([], 10.0)
    with pytest.raises(ValueError):
        segmented_cost([2.0], 0.0)
    with pytest.raises(ValueError):
        segmented_cost([2.0, 3.0], 10.0, weights=[1.0])
