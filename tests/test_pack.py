"""Shard writer round trip and the pseudo-label filters."""

import math

import numpy as np
import pytest

from whisper_distill.data.pack import (
    PAD_TOKEN,
    WHISPER_FLOOR_OFFSET,
    ShardWriter,
    open_shards,
    whisper_floor,
)
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


def test_round_trip_preserves_content_and_tokens(tmp_path):
    index, shards = _write(tmp_path)
    assert len(index.records) == 7
    r = index.records[5]
    mels, tokens = shards[r.shard]
    assert mels[r.row].shape == (80, 1000)
    assert np.allclose(mels[r.row][:, :300], 5.0)
    assert list(tokens[r.row][:4]) == [1, 2, 3, 5]
    assert tokens[r.row][4] == PAD_TOKEN


def test_whisper_floor_is_two_below_the_maximum():
    """Whisper normalises log-mel as (log10+4)/4 after flooring at log.max()-8, so the
    floor sits exactly 8/4 = 2.0 below the normalised maximum. Exact for any input."""
    mel = np.array([[-0.5, 1.25, 0.0]], dtype=np.float32)
    assert whisper_floor(mel) == pytest.approx(1.25 - WHISPER_FLOOR_OFFSET)


def test_short_mel_is_padded_with_the_whisper_floor_not_zero(tmp_path):
    """Zero-padding would feed a FROZEN encoder a value it never saw in training, and a
    frozen encoder cannot adapt. This is the bug the previous assertion enshrined."""
    mel = np.full((80, 300), 0.75, dtype=np.float32)
    with ShardWriter(tmp_path, n_mels=80, n_frames=1000, max_tokens=8) as w:
        w.add("c0", mel, [1, 2])
    index, shards = open_shards(tmp_path)
    row = shards[0][0][0]
    assert np.allclose(row[:, :300], 0.75)
    tail = row[:, 300:]
    assert not np.any(tail == 0.0), "padding must never be zero"
    assert np.allclose(tail, 0.75 - WHISPER_FLOOR_OFFSET, atol=1e-3)


def test_full_width_mel_is_stored_untouched(tmp_path):
    """The preferred path: slice the teacher's own feature tensor, padding included."""
    mel = np.full((80, 1000), 0.4, dtype=np.float32)
    mel[:, 600:] = -0.83                     # the teacher's own floor region
    with ShardWriter(tmp_path, n_mels=80, n_frames=1000, max_tokens=8) as w:
        w.add("c0", mel, [1], speech_frames=600)
    index, shards = open_shards(tmp_path)
    assert np.allclose(shards[0][0][0], mel, atol=1e-3)
    assert index.records[0].n_frames == 600  # metadata records real speech, not width


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


# ------------------------------------------------------- version-agnostic audio decoding
def test_datasets_3x_dict_is_decoded_and_resampled():
    from whisper_distill.data.audio_io import decode_audio_field

    wav, how = decode_audio_field(
        {"array": np.ones(8000, dtype=np.float32), "sampling_rate": 8000}
    )
    assert how == "dict_array"
    assert len(wav) == 16000  # 8 kHz -> 16 kHz


def test_datasets_4x_audiodecoder_is_decoded_and_downmixed():
    """datasets 4.0 returns a torchcodec AudioDecoder, not a dict."""
    from whisper_distill.data.audio_io import decode_audio_field

    class _Samples:
        data = np.ones((2, 4000), dtype=np.float32)  # (channels, n)
        sample_rate = 16000

    class _Decoder:
        def get_all_samples(self):
            return _Samples()

    wav, how = decode_audio_field(_Decoder())
    assert how == "audio_decoder"
    assert wav.shape == (4000,)


def test_unknown_audio_field_raises_rather_than_guessing():
    from whisper_distill.data.audio_io import decode_audio_field

    with pytest.raises(TypeError, match="unrecognised audio field"):
        decode_audio_field(object())


def test_mono_downmix_handles_both_axis_orders():
    """soundfile gives (n, channels); torchcodec gives (channels, n)."""
    from whisper_distill.data.audio_io import to_mono

    assert to_mono(np.ones((2, 500))).shape == (500,)
    assert to_mono(np.ones((500, 2))).shape == (500,)
    assert to_mono(np.ones(500)).shape == (500,)
    with pytest.raises(ValueError):
        to_mono(np.ones((2, 2, 2)))


def test_schema_probe_names_the_missing_column():
    """An empty transcript silently disables step 2's WER filter, so fail loudly."""
    from whisper_distill.data.audio_io import probe_schema

    row = {"audio": {"array": np.zeros(1), "sampling_rate": 16000}, "transcript": "hi"}
    assert "array" in probe_schema(row)
    with pytest.raises(KeyError, match="transcript"):
        probe_schema({"audio": row["audio"]})


def test_resample_is_a_noop_at_the_target_rate():
    from whisper_distill.data.audio_io import resample

    x = np.arange(100, dtype=np.float32)
    assert np.array_equal(resample(x, 16000, 16000), x)


# --------------------------------------------------- schema discovery for unknown corpora
def test_transcript_key_prefers_a_populated_column():
    """A corpus can carry an empty `text` beside a filled `verbatim`. Picking the empty one
    silently disables the WER filter, which looks like success."""
    from whisper_distill.data.audio_io import find_transcript_key

    row = {"text": "", "verbatim": "meeting 4 baje hai", "speaker_id": "s1"}
    assert find_transcript_key(row) == "verbatim"


def test_transcript_key_follows_the_preference_order():
    from whisper_distill.data.audio_io import find_transcript_key

    assert find_transcript_key({"text": "a", "transcript": "b"}) == "transcript"
    assert find_transcript_key({"sentence": "a"}) == "sentence"
    assert find_transcript_key({"speaker": "x"}) is None


def test_audio_key_is_found_by_shape_when_the_name_is_unknown():
    """IndicVoices publishes no schema, so name-based lookup cannot be the only path."""
    from whisper_distill.data.audio_io import find_audio_key

    class _Decoder:
        def get_all_samples(self):  # pragma: no cover - shape marker only
            raise NotImplementedError

    assert find_audio_key({"oddly_named": _Decoder()}) == "oddly_named"
    assert find_audio_key({"wav": {"bytes": b"x"}}) == "wav"
    assert find_audio_key({"audio": {"array": np.zeros(1), "sampling_rate": 16000}}) == "audio"
    assert find_audio_key({"speaker_id": "s1"}) is None


def test_describe_row_reports_both_keys_and_the_leftovers():
    from whisper_distill.data.audio_io import describe_row

    out = describe_row({
        "audio": {"array": np.zeros(1), "sampling_rate": 16000},
        "transcript": "hi",
        "gender": "female",
    })
    assert "'audio'" in out and "'transcript'" in out and "gender" in out
