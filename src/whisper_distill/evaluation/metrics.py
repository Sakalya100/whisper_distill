"""Scoring. Pure Python and stdlib only -- runs in a free CPU session.

Four things get reported, because average WER hides every failure this project cares about:

  1. WER and CER, split by code-mixing density
  2. Punctuated WER, not only normalised WER -- dictation output needs punctuation, so
     scoring only the normalised form measures the wrong model
  3. Entity accuracy on numbers, dates and times (see `entities.py`)
  4. On-device RTF, peak RAM, battery drain -- measured on a phone, not here
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

_PUNCT = re.compile(r"[^\w\sऀ-ॿ]", re.UNICODE)
_WS = re.compile(r"\s+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

#: High-frequency English tokens that show up in Hinglish dictation. Used only to estimate
#: code-mix density for bucketing; the eval-set manifest carries a hand-checked bucket and
#: that is what results are reported against. This is a triage aid, not ground truth.
DEFAULT_EN_MARKERS = frozenset("""
meeting call slack email message reminder office project client deadline update
report file folder link photo video download upload login password account
order delivery payment invoice discount offer cancel confirm book ticket
doctor appointment medicine test result school class exam homework
today tomorrow yesterday morning evening night weekend monday tuesday wednesday
thursday friday saturday sunday minute hour week month year
please thanks sorry okay ok yes no maybe done pending urgent important
add remove edit delete send share save open close start stop check
laptop phone charger battery wifi internet app screen camera
""".split())


# ======================================================================================
# Edit distance
# ======================================================================================

def _levenshtein(ref: Sequence, hyp: Sequence) -> int:
    """Standard DP edit distance, two rows. O(len(ref)) memory."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(ref) + 1))
    for j, h in enumerate(hyp, start=1):
        cur = [j] + [0] * len(ref)
        for i, r in enumerate(ref, start=1):
            cur[i] = min(
                prev[i] + 1,          # deletion from hyp perspective
                cur[i - 1] + 1,       # insertion
                prev[i - 1] + (r != h),  # substitution
            )
        prev = cur
    return prev[-1]


# ======================================================================================
# Normalisation
# ======================================================================================

def normalise_for_wer(text: str, *, strip_punctuation: bool = True) -> str:
    """Whisper-style light normalisation: NFKC, casefold, collapse whitespace.

    `strip_punctuation=False` gives the **punctuated** form. Both are needed: the
    normalised number is comparable with published Indic ASR results, the punctuated
    number is the one that reflects what a dictation user sees.
    """
    t = unicodedata.normalize("NFKC", text or "").casefold()
    if strip_punctuation:
        t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def wer(reference: str, hypothesis: str, *, punctuated: bool = False) -> tuple[int, int]:
    """Returns ``(errors, reference_length)`` so callers can pool a corpus correctly.

    Per-utterance WERs must never be averaged -- short utterances would dominate. Pool the
    numerators and denominators, which is what `report` does.
    """
    ref = normalise_for_wer(reference, strip_punctuation=not punctuated).split()
    hyp = normalise_for_wer(hypothesis, strip_punctuation=not punctuated).split()
    return _levenshtein(ref, hyp), len(ref)


def cer(reference: str, hypothesis: str) -> tuple[int, int]:
    ref = normalise_for_wer(reference).replace(" ", "")
    hyp = normalise_for_wer(hypothesis).replace(" ", "")
    return _levenshtein(ref, hyp), len(ref)


# ======================================================================================
# Code-mixing
# ======================================================================================

def code_mix_density(text: str, en_markers: Iterable[str] | None = None) -> float:
    """Rough fraction of tokens that are English.

    Two signals, because romanised Hinglish gives script no purchase: if the text contains
    Devanagari, non-Devanagari word tokens count as English; otherwise fall back to a
    marker wordlist. Both are approximations -- the eval-set manifest carries a
    hand-checked bucket and that is what gets reported.
    """
    markers = frozenset(m.casefold() for m in (en_markers or DEFAULT_EN_MARKERS))
    tokens = normalise_for_wer(text).split()
    if not tokens:
        return 0.0
    if _DEVANAGARI.search(text or ""):
        english = sum(1 for t in tokens if not _DEVANAGARI.search(t))
    else:
        english = sum(1 for t in tokens if t in markers)
    return english / len(tokens)


def code_mix_bucket(text: str, en_markers: Iterable[str] | None = None) -> str:
    """``hindi_dominant`` < 0.15 <= ``balanced`` < 0.40 <= ``dense``."""
    d = code_mix_density(text, en_markers)
    if d < 0.15:
        return "hindi_dominant"
    if d < 0.40:
        return "balanced"
    return "dense"


# ======================================================================================
# Reporting
# ======================================================================================

@dataclass
class _Acc:
    err: int = 0
    total: int = 0

    def add(self, err: int, total: int) -> None:
        self.err += err
        self.total += total

    @property
    def rate(self) -> float:
        return self.err / self.total if self.total else 0.0


@dataclass
class Report:
    n_utterances: int = 0
    overall: dict[str, float] = field(default_factory=dict)
    by_bucket: dict[str, dict[str, float]] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"utterances: {self.n_utterances}", ""]
        w = max((len(k) for k in self.overall), default=8)
        for k, v in self.overall.items():
            lines.append(f"  {k:<{w}}  {v * 100:6.2f}%")
        if self.by_bucket:
            lines += ["", "by code-mix density:"]
            for bucket in ("hindi_dominant", "balanced", "dense"):
                if bucket not in self.by_bucket:
                    continue
                m = self.by_bucket[bucket]
                lines.append(
                    f"  {bucket:<15} n={int(m['n']):<4} "
                    f"WER {m['wer'] * 100:6.2f}%  "
                    f"pWER {m['punctuated_wer'] * 100:6.2f}%  "
                    f"CER {m['cer'] * 100:6.2f}%"
                )
        return "\n".join(lines)


def report(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    buckets: Sequence[str] | None = None,
) -> Report:
    """Pool errors over a corpus and split by code-mix bucket.

    `buckets` should be the hand-checked values from the eval-set manifest. When omitted
    they are estimated with `code_mix_bucket`, which is a triage aid only.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"{len(references)} references vs {len(hypotheses)} hypotheses"
        )
    if buckets is not None and len(buckets) != len(references):
        raise ValueError("buckets must align with references")

    acc = {"wer": _Acc(), "punctuated_wer": _Acc(), "cer": _Acc()}
    per_bucket: dict[str, dict[str, _Acc]] = {}

    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        b = buckets[i] if buckets is not None else code_mix_bucket(ref)
        slot = per_bucket.setdefault(
            b, {"wer": _Acc(), "punctuated_wer": _Acc(), "cer": _Acc(), "n": _Acc()}
        )
        slot["n"].add(0, 1)
        for name, (e, t) in (
            ("wer", wer(ref, hyp)),
            ("punctuated_wer", wer(ref, hyp, punctuated=True)),
            ("cer", cer(ref, hyp)),
        ):
            acc[name].add(e, t)
            slot[name].add(e, t)

    return Report(
        n_utterances=len(references),
        overall={k: v.rate for k, v in acc.items()},
        by_bucket={
            b: {
                "n": float(s["n"].total),
                **{k: s[k].rate for k in ("wer", "punctuated_wer", "cer")},
            }
            for b, s in per_bucket.items()
        },
    )
