"""Inverse text normalisation. Users judge dictation on these, not on average WER."""

import pytest

from whisper_distill.itn import normalise


@pytest.mark.parametrize(
    "spoken,written",
    [
        # The Gate 1 canonical case: Latin-script English must pass through untouched.
        ("meeting chaar baje hai, Slack pe ping karo",
         "meeting 4 baje hai, Slack pe ping karo"),
        # Fractional-hour prefixes -- the reason times run before numbers.
        ("saade chaar baje call karna", "4:30 baje call karna"),
        ("sawa paanch baje nikalna hai", "5:15 baje nikalna hai"),
        ("paune paanch baje pahunchna", "4:45 baje pahunchna"),
        ("chaar bajkar bees minute par alarm", "4:20 baje par alarm"),
        ("shaam saat baje dinner", "7 pm baje dinner"),
        # Compositional numbers.
        ("do hazaar bees mein hua tha", "2020 mein hua tha"),
        ("teen sau paanch rupaye", "₹305"),
        ("bees percent discount", "20% discount"),
        ("pandrah August ko chhutti hai", "15 August ko chhutti hai"),
        # Bare counts still convert; the trailing noun is left alone.
        ("do litre doodh aur paanch kilo aata", "2 litre doodh aur 5 kilo aata"),
        # Devanagari input.
        ("चार बजे मीटिंग", "4 baje मीटिंग"),
    ],
)
def test_known_conversions(spoken, written):
    assert normalise(spoken) == written


@pytest.mark.parametrize("text", [
    "Slack pe reply karna mat bhoolna",
    "kal office jaana hai",
    "",
    "   ",
])
def test_leaves_number_free_text_alone(text):
    assert normalise(text) == text


def test_paune_wraps_at_one_oclock():
    """'paune ek baje' is 12:45, not 0:45."""
    assert normalise("paune ek baje") == "12:45 baje"


def test_bare_number_is_not_read_as_a_time():
    """Without 'baje' or a fractional prefix, a number stays a number."""
    assert normalise("chaar log aaye the") == "4 log aaye the"


def test_trailing_punctuation_survives_conversion():
    assert normalise("chaar baje.") == "4 baje."
    assert normalise("bees percent!") == "20%!"
