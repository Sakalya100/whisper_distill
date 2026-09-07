"""Pure-Python surgery helpers. No torch required."""

import pytest

from whisper_distill.modeling.surgery import (
    assert_labels_within_vocabulary,
    maximally_spaced_indices,
    select_vocabulary,
)


def test_endpoints_are_always_kept():
    """Distil-Whisper's 2-layer case must reduce to first-and-last."""
    assert maximally_spaced_indices(32, 2) == [0, 31]
    assert maximally_spaced_indices(12, 2) == [0, 11]


def test_twelve_to_four_is_evenly_spread():
    assert maximally_spaced_indices(12, 4) == [0, 4, 7, 11]


def test_indices_are_strictly_increasing_and_in_range():
    for n_from in range(1, 40):
        for n_to in range(1, n_from + 1):
            idx = maximally_spaced_indices(n_from, n_to)
            assert len(idx) == n_to
            assert idx == sorted(set(idx)), (n_from, n_to, idx)
            assert 0 <= idx[0] and idx[-1] <= n_from - 1, (n_from, n_to, idx)


def test_identity_when_keeping_everything():
    assert maximally_spaced_indices(12, 12) == list(range(12))


def test_rejects_impossible_requests():
    with pytest.raises(ValueError):
        maximally_spaced_indices(4, 5)
    with pytest.raises(ValueError):
        maximally_spaced_indices(4, 0)


def test_mandatory_tokens_survive_regardless_of_frequency():
    """Dropping BOS/EOS or the language token yields a model that cannot start decoding."""
    counts = {i: 1000 for i in range(100, 200)}  # all far more frequent than the specials
    keep = select_vocabulary(counts, target_size=10, always_keep=[0, 1, 2])
    assert {0, 1, 2} <= set(keep)
    assert len(keep) == 10


def test_selection_prefers_frequent_tokens_and_is_sorted():
    counts = {10: 5, 11: 99, 12: 1, 13: 50}
    keep = select_vocabulary(counts, target_size=4, always_keep=[0, 1])
    assert keep == [0, 1, 11, 13]  # 12 and 10 are the rarest, dropped
    assert keep == sorted(keep)    # order-preserving remap depends on this


def test_budget_smaller_than_mandatory_set_is_an_error():
    with pytest.raises(ValueError):
        select_vocabulary({}, target_size=2, always_keep=[0, 1, 2])


def test_out_of_vocabulary_labels_are_rejected_not_clamped():
    """Silently clamping is how pruning produces a model that trains and outputs nonsense."""
    mapping = {5: 0, 7: 1}
    assert_labels_within_vocabulary([5, 7, -100], mapping)  # -100 is the pad, ignored
    with pytest.raises(ValueError, match="outside the pruned vocabulary"):
        assert_labels_within_vocabulary([5, 9], mapping)
