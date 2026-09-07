"""WER/CER pooling, punctuated vs normalised scoring, entity accuracy."""

import pytest

from whisper_distill.evaluation.entities import Entity, score_entities
from whisper_distill.evaluation.metrics import cer, code_mix_bucket, report, wer


def test_wer_counts_each_error_type():
    assert wer("a b c", "a b c") == (0, 3)
    assert wer("a b c", "a x c") == (1, 3)   # substitution
    assert wer("a b c", "a b") == (1, 3)     # deletion
    assert wer("a b c", "a b c d") == (1, 3) # insertion


def test_wer_returns_errors_and_length_not_a_rate():
    """Per-utterance rates must never be averaged -- short utterances would dominate."""
    errors, length = wer("ek do teen chaar", "ek do teen")
    assert (errors, length) == (1, 4)


def test_punctuation_only_counts_in_the_punctuated_variant():
    """Dictation needs punctuation, so scoring only the normalised form measures the
    wrong model."""
    assert wer("hello, world", "hello world")[0] == 0
    assert wer("hello, world", "hello world", punctuated=True)[0] == 1


def test_empty_reference_reports_zero_length():
    assert wer("", "anything") == (1, 0)


def test_cer_ignores_spacing():
    assert cer("chaar baje", "chaarbaje")[0] == 0


def test_report_pools_rather_than_averages():
    """Two utterances, one perfect and one wholly wrong, of very different lengths."""
    refs = ["a", "b c d e f g h i j k"]
    hyps = ["a", "x x x x x x x x x x"]
    r = report(refs, hyps)
    assert r.n_utterances == 2
    # Pooled: 10 errors over 11 reference words. An average of per-utterance rates
    # would give 0.5, which would be wrong.
    assert r.overall["wer"] == pytest.approx(10 / 11)


def test_report_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        report(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        report(["a"], ["a"], buckets=["dense", "dense"])


def test_report_splits_by_bucket():
    r = report(["a b", "c d"], ["a b", "x d"], buckets=["dense", "hindi_dominant"])
    assert r.by_bucket["dense"]["wer"] == 0.0
    assert r.by_bucket["hindi_dominant"]["wer"] == 0.5


def test_code_mix_bucket_separates_registers():
    assert code_mix_bucket("kal subah jaana hai") == "hindi_dominant"
    assert code_mix_bucket("meeting call email project deadline update") == "dense"


def test_entity_accuracy_is_scored_on_the_punctuated_surface():
    """'4:30' and '430' are different answers to a user."""
    ents = [[Entity("time", "4:30 baje", "16:30")]]
    assert score_entities(ents, ["call 4:30 baje karna"]).overall == 1.0
    assert score_entities(ents, ["call 430 baje karna"]).overall == 0.0


def test_entity_types_are_validated():
    with pytest.raises(ValueError, match="unknown entity type"):
        Entity("vibe", "whatever")
