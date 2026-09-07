"""Rule-based inverse text normalisation for Hinglish dictation.

Do not make a 132M model learn that "saade chaar baje" is "4:30". Users judge dictation on
numbers, dates and times, not on average WER -- get "chaar baje" -> "4 baje" wrong and the
notes feel broken at 8% WER. This runs after the model, on CPU, and improves perceived
quality more than three points of WER.

Design: three ordered passes over whitespace tokens.

  1. Time expressions first  -- "saade chaar baje" must not become "saade 4 baje".
  2. Number runs             -- compositional, so "do hazaar bees" -> 2020.
  3. Trailing units          -- percent, currency.

Coverage is deliberately partial and honest about it: see `COVERAGE_GAPS`. Extending the
lexicon is cheap; guessing at irregular Hindi decades is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ======================================================================================
# Lexicon
# ======================================================================================

#: Units, 0-20. Latin transliteration is intentionally multi-spelled -- dictation output
#: is not orthographically stable, and the teacher's spelling is whatever it learned.
_UNITS: dict[str, int] = {
    "zero": 0, "shunya": 0, "sifar": 0, "शून्य": 0,
    "one": 1, "ek": 1, "एक": 1,
    "two": 2, "do": 2, "दो": 2,
    "three": 3, "teen": 3, "tin": 3, "तीन": 3,
    "four": 4, "char": 4, "chaar": 4, "चार": 4,
    "five": 5, "panch": 5, "paanch": 5, "पांच": 5, "पाँच": 5,
    "six": 6, "che": 6, "chhe": 6, "chah": 6, "chhah": 6, "छह": 6, "छे": 6,
    "seven": 7, "sat": 7, "saat": 7, "सात": 7,
    "eight": 8, "ath": 8, "aath": 8, "आठ": 8,
    "nine": 9, "nau": 9, "no": 9, "नौ": 9,
    "ten": 10, "das": 10, "dus": 10, "दस": 10,
    "eleven": 11, "gyarah": 11, "gyara": 11, "ग्यारह": 11,
    "twelve": 12, "barah": 12, "bara": 12, "बारह": 12,
    "thirteen": 13, "terah": 13, "तेरह": 13,
    "fourteen": 14, "chaudah": 14, "चौदह": 14,
    "fifteen": 15, "pandrah": 15, "pandra": 15, "पंद्रह": 15,
    "sixteen": 16, "solah": 16, "सोलह": 16,
    "seventeen": 17, "satrah": 17, "सत्रह": 17,
    "eighteen": 18, "atharah": 18, "अठारह": 18,
    "nineteen": 19, "unnis": 19, "उन्नीस": 19,
    "twenty": 20, "bees": 20, "bis": 20, "बीस": 20,
}

#: High-frequency irregular decades and mid-decade forms. Hindi 21-99 has no clean rule,
#: so this is a curated set rather than a generated one.
_TENS_AND_IRREGULARS: dict[str, int] = {
    "ikkis": 21, "इक्कीस": 21,
    "baees": 22, "bais": 22, "बाईस": 22,
    "teies": 23, "teis": 23, "तेईस": 23,
    "chaubis": 24, "चौबीस": 24,
    "pachees": 25, "pachchis": 25, "पच्चीस": 25, "twentyfive": 25,
    "tees": 30, "tis": 30, "तीस": 30, "thirty": 30,
    "paintees": 35, "पैंतीस": 35, "thirtyfive": 35,
    "chalees": 40, "chalis": 40, "चालीस": 40, "forty": 40,
    "pachaas": 50, "pachas": 50, "पचास": 50, "fifty": 50,
    "saath": 60, "साठ": 60, "sixty": 60,
    "sattar": 70, "सत्तर": 70, "seventy": 70,
    "pachhattar": 75, "पचहत्तर": 75, "seventyfive": 75,
    "assi": 80, "अस्सी": 80, "eighty": 80,
    "nabbe": 90, "नब्बे": 90, "ninety": 90,
}

#: Multiplicative scales, smallest first for the composition loop.
_SCALES: dict[str, int] = {
    "hundred": 100, "sau": 100, "सौ": 100,
    "thousand": 1_000, "hazaar": 1_000, "hazar": 1_000, "हज़ार": 1_000, "हजार": 1_000,
    "lakh": 100_000, "lac": 100_000, "लाख": 100_000,
    "million": 1_000_000,
    "crore": 10_000_000, "karod": 10_000_000, "करोड़": 10_000_000,
}

_CARDINALS: dict[str, int] = {**_UNITS, **_TENS_AND_IRREGULARS}

#: Fractional-hour prefixes. These are the reason time has to run before numbers.
_HOUR_OFFSET: dict[str, int] = {
    "saade": 30, "sade": 30, "साढ़े": 30,   # half past
    "sawa": 15, "sava": 15, "सवा": 15,      # quarter past
    "paune": -15, "पौने": -15,              # quarter to  -> previous hour + 45
}

_MERIDIEM: dict[str, str] = {
    "subah": "am", "सुबह": "am",
    "dopahar": "pm", "दोपहर": "pm",
    "shaam": "pm", "sham": "pm", "शाम": "pm",
    "raat": "pm", "रात": "pm",
}

_OCLOCK = {"baje", "bajay", "baja", "बजे", "बजकर"}
_PAST_MARKER = {"bajkar", "बजकर"}
_MINUTE_WORD = {"minute", "minat", "min", "मिनट"}
_PERCENT = {"percent", "pratishat", "प्रतिशत", "फ़ीसदी", "fisadi"}
_CURRENCY = {"rupaye", "rupees", "rupya", "rs", "रुपये", "रुपए"}

_MONTHS = {
    m.lower(): m for m in (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    )
}
_MONTHS.update({
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "jun": "June", "jul": "July", "aug": "August", "sep": "September",
    "sept": "September", "oct": "October", "nov": "November", "dec": "December",
})

COVERAGE_GAPS = (
    "Hindi 21-99 is only partially covered -- irregular decades are curated, not generated. "
    "Ordinals (pehla, doosra), fractions beyond quarter/half, and relative dates "
    "(kal, parson, agle hafte) are not handled. Add cases from real eval-set failures "
    "rather than speculatively.",
)

_TOKEN_RE = re.compile(r"(\s+)")
_TRAILING_PUNCT = re.compile(r"^(.*?)([,.!?;:।]*)$", re.DOTALL)


# ======================================================================================
# Helpers
# ======================================================================================

def _split(token: str) -> tuple[str, str]:
    """Strip trailing punctuation so lexicon lookups still hit on 'baje,'."""
    m = _TRAILING_PUNCT.match(token)
    return (m.group(1), m.group(2)) if m else (token, "")


def _key(token: str) -> str:
    return _split(token)[0].lower().strip("\"'()")


def _as_int(token: str) -> int | None:
    """Word or digit string to int, or None."""
    k = _key(token)
    if not k:
        return None
    if k.isdigit():
        return int(k)
    return _CARDINALS.get(k)


def _compose(values: list[int]) -> int:
    """Fold a run of cardinal/scale values into one number.

    Handles "do hazaar bees" (2 x 1000 + 20 = 2020) and "teen sau paanch" (305), and the
    additive Hindi/English pattern "bees paanch" is *not* treated as 25 -- that reading is
    ambiguous in dictation and guessing it wrong is worse than leaving it alone.
    """
    total = 0
    current = 0
    for v in values:
        if v >= 100:
            current = (current or 1) * v
            total += current
            current = 0
        else:
            current += v
    return total + current


def _fmt_time(hour: int, minute: int, meridiem: str | None) -> str:
    hour = hour % 24 if hour > 12 else hour
    stamp = f"{hour}:{minute:02d}" if minute else f"{hour}"
    return f"{stamp} {meridiem}" if meridiem else stamp


# ======================================================================================
# Passes
# ======================================================================================

@dataclass
class HinglishITN:
    """Ordered rule passes. Stateless; safe to reuse across a whole eval set."""

    keep_baje: bool = True
    """Keep the Hindi word 'baje' after a converted time. Users write "4 baje", not "4:00"."""

    def __call__(self, text: str) -> str:
        return self.apply(text)

    def apply(self, text: str) -> str:
        if not text or not text.strip():
            return text
        tokens = text.split()
        tokens = self._times(tokens)
        tokens = self._numbers(tokens)
        tokens = self._units(tokens)
        return " ".join(tokens)

    # ---------------------------------------------------------------- pass 1: times
    def _times(self, tokens: list[str]) -> list[str]:
        out: list[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            consumed = self._match_time(tokens, i)
            if consumed:
                text, width = consumed
                # Carry any punctuation that closed the matched span.
                _, punct = _split(tokens[i + width - 1])
                out.append(text + punct)
                i += width
                continue
            out.append(tokens[i])
            i += 1
        return out

    def _match_time(self, tokens: list[str], i: int) -> tuple[str, int] | None:
        n = len(tokens)
        meridiem = None
        start = i
        # Optional leading meridiem: "shaam saade paanch baje"
        if _key(tokens[i]) in _MERIDIEM:
            if i + 1 >= n:
                return None
            meridiem = _MERIDIEM[_key(tokens[i])]
            i += 1

        offset_key = _key(tokens[i]) if i < n else ""
        offset = _HOUR_OFFSET.get(offset_key)
        if offset is not None:
            i += 1

        hour = _as_int(tokens[i]) if i < n else None
        if hour is None or not 1 <= hour <= 24:
            return None
        i += 1

        # "chaar bajkar bees minute" -> 4:20
        if i < n and _key(tokens[i]) in _PAST_MARKER:
            minute = _as_int(tokens[i + 1]) if i + 1 < n else None
            if minute is not None and 0 <= minute < 60:
                j = i + 2
                if j < n and _key(tokens[j]) in _MINUTE_WORD:
                    j += 1
                if offset is not None:
                    return None  # "saade chaar bajkar" is not a real construction
                tail = self._tail(tokens, j)
                stamp = _fmt_time(hour, minute, meridiem or tail[0])
                suffix = " baje" if self.keep_baje else ""
                return stamp + suffix, (tail[1] - start)
            return None

        if i < n and _key(tokens[i]) in _OCLOCK:
            i += 1
        elif offset is None:
            return None  # a bare number is not a time; leave it to pass 2

        if offset is None:
            minute, out_hour = 0, hour
        elif offset < 0:
            minute, out_hour = 45, hour - 1 if hour > 1 else 12
        else:
            minute, out_hour = offset, hour

        tail_meridiem, end = self._tail(tokens, i)
        stamp = _fmt_time(out_hour, minute, meridiem or tail_meridiem)
        suffix = " baje" if self.keep_baje else ""
        return stamp + suffix, (end - start)

    @staticmethod
    def _tail(tokens: list[str], i: int) -> tuple[str | None, int]:
        """Absorb a trailing meridiem word, returning it and the new cursor."""
        if i < len(tokens) and _key(tokens[i]) in _MERIDIEM:
            return _MERIDIEM[_key(tokens[i])], i + 1
        return None, i

    # -------------------------------------------------------------- pass 2: numbers
    def _numbers(self, tokens: list[str]) -> list[str]:
        out: list[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            run: list[int] = []
            j = i
            punct = ""
            while j < n:
                k = _key(tokens[j])
                if k in _SCALES:
                    run.append(_SCALES[k])
                elif (v := _as_int(tokens[j])) is not None and not k.isdigit():
                    run.append(v)
                else:
                    break
                punct = _split(tokens[j])[1]
                j += 1
            if run:
                # A lone scale word ("sau rupaye" with no multiplier) still reads as 100.
                out.append(str(_compose(run)) + punct)
                i = j
            else:
                out.append(tokens[i])
                i += 1
        return out

    # ---------------------------------------------------------------- pass 3: units
    def _units(self, tokens: list[str]) -> list[str]:
        out: list[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            cur, punct = _split(tokens[i])
            nxt = _key(tokens[i + 1]) if i + 1 < n else ""

            if cur.isdigit() and nxt in _PERCENT:
                _, npunct = _split(tokens[i + 1])
                out.append(f"{cur}%{npunct}")
                i += 2
                continue
            if cur.isdigit() and nxt in _CURRENCY:
                _, npunct = _split(tokens[i + 1])
                out.append(f"₹{cur}{npunct}")
                i += 2
                continue
            if cur.isdigit() and nxt in _MONTHS:
                _, npunct = _split(tokens[i + 1])
                out.append(f"{cur}")
                out.append(_MONTHS[nxt] + npunct)
                i += 2
                continue
            out.append(tokens[i])
            i += 1
        return out


_DEFAULT = HinglishITN()


def normalise(text: str) -> str:
    """Module-level convenience using default settings."""
    return _DEFAULT.apply(text)
