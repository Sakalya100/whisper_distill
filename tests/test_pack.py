"""Shard writer round trip and the pseudo-label filters."""

import math

import numpy as np
import pytest

from whisper_distill.data.pack import PAD_TOKEN, ShardWriter, open_shards
from whisper_distill.data.segment import Segment, merge_to_window
from whisper_distill.labeling.filters import (
    calibrate_entropy_threshold,
    mean_token_entropy,
    passes_wer_filter,
)


# ------------------------------------------------------------------ segmentation
def test_merge_respects_the_ten_second_ceiling():
    """The ceiling is what makes the student mel a prefix of the teacher's padded mel."""
    speech = [Segment(0, 4), Segment(4.2, 8), Segment(8.1, 14)]
    out = merge_to_window(speech, max_seconds=10, min_seconds=1)
    assert out, "everything was dropped"
    assert all(s.duration_s <= 10 + 1e-6 for s in out)


def test_long_region_is_split_not_dropped():
    out = merge_to_window([Segment(0, 25)], max_seconds=10, min_seconds=1)
    assert [round(s.duration_s, 3) for s in out] == [10.0, 10.0, 5.0]


def test_short_fragments_are_dropped():
    out = merge_to_window([Segment(0, 0.3), Segment(5, 5.2)], max_seconds=10, min_seconds=1)
    assert out == []


def test_wide_gaps_are_not_merged():
    out = merge_to_window(
        [Segment(0, 2), Segment(8, 10)], max_seconds=10, min_seconds=1, max_gap_s=0.6
    )
    assert len(out) == 2


def test_unsorted_input_is_handled():
    out = merge_to_window([Segment(5, 7), Segment(0, 2)], max_seconds=10, min_seconds=1)
    assert [s.start_s for s in out] == sorted(s.start_s for s in out)


# ------------------------------------------------------------------ shard packing
def _write(tmp_path, n=7):
    with ShardWriter(
        tmp_path, n_mels=80, n_frames=1000, max_tokens=16, shard_target_bytes=400_000
    ) as w:
        for i in range(n):
            mel = np.full((80, 300 + i), float(i), dtype=np.float32)
            w.add(f"c{i}", mel, [1, 2, 3, i], source="test",
                  duration_s=3.0, teacher_wer=0.05 * i, text=f"clip {i}")
    return open_shards(tmp_path)


def test_round_trip_preserves_content_and_pads_the_rest(tmp_path):
    index, shards = _write(tmp_path)
    assert len(index.records) == 7
    r = index.records[5]
    mels, tokens = shards[r.shard]
    assert mels[r.row].shape == (80, 1000)
    assert np.allclose(mels[r.row][:, : r.n_frames], 5.0)
    assert np.all(mels[r.row][:, r.n_frames :] == 0), "padding must be zero, as Whisper's is"
    assert list(tokens[r.row][:4]) == [1, 2, 3, 5]
    assert tokens[r.row][4] == PAD_TOKEN


def test_rolls_to_multiple_shards(tmp_path):
    _, shards = _write(tmp_path)
    assert len(shards) > 1, "byte target should have forced a roll"


def test_wer_filter_applies_on_read(tmp_path):
    index, _ = _write(tmp_path)
    assert len(index.filtered(max_teacher_wer=0.20)) == 5  # 0.00 .. 0.20 inclusive


def test_over_long_mel_is_rejected_with_a_pointer_to_segmentation(tmp_path):
    with ShardWriter(tmp_path, n_mels=80, n_frames=1000) as w:
        with pytest.raises(ValueError, match="Segment to <=10 s"):
            w.add("bad", np.zeros((80, 1500), dtype=np.float32), [1])


def test_wrong_mel_bin_count_is_rejected(tmp_path):
    with ShardWriter(tmp_path, n_mels=80, n_frames=1000) as w:
        with pytest.raises(ValueError):
            w.add("bad", np.zeros((128, 500), dtype=np.float32), [1])


# ------------------------------------------------------------------ filters
def test_wer_filter_keeps_a_punctuated_label_against_a_normalised_reference():
    """This is the whole point of pseudo-labelling: the label has punctuation and casing
    the human reference lacks, and must not be rejected for having them."""
    assert passes_wer_filter(
        "meeting chaar baje hai slack pe ping karo",
        "Meeting chaar baje hai, Slack pe ping karo.",
    )


def test_wer_filter_rejects_a_hallucination():
    assert not passes_wer_filter("doodh aur aata lena hai", "thank you for watching")


def test_wer_filter_rejects_an_empty_reference():
    assert not passes_wer_filter("", "anything")


def test_entropy_is_lower_for_a_confident_distribution():
    peaked = [[math.log(0.97), math.log(0.02), math.log(0.01)]]
    flat = [[math.log(1 / 3)] * 3]
    assert mean_token_entropy(peaked) < mean_token_entropy(flat)
    assert mean_token_entropy([]) == 0.0


def test_entropy_threshold_is_calibrated_to_the_wer_keep_rate():
    """Pick the cutoff that reproduces the WER filter's keep rate, don't guess a number."""
    entropies = [0.1, 0.5, 0.9, 2.0, 3.0]
    keeps = [True, True, True, False, False]
    assert calibrate_entropy_threshold(entropies, keeps) == 0.9
