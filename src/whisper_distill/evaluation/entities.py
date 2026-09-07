"""Entity accuracy on numbers, dates and times.

This is what gates perceived quality. A user judges dictation on whether "chaar baje"
became "4 baje", not on average WER -- get it wrong and the notes feel broken at 8% WER.
Scored separately from WER so a regression here cannot hide inside a good average.

Entities come from the eval-set manifest (`eval_set/README.md`), hand-annotated with a
`surface` string and a canonical `value`. Scoring is surface presence after ITN, because
that is what the user actually reads.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from whisper_distill.evaluation.metrics import normalise_for_wer

ENTITY_TYPES = ("number", "time", "date", "currency", "percent")


@dataclass(frozen=True)
class Entity:
    type: str
    surface: str
    value: str = ""

    def __post_init__(self) -> None:
        if self.type not in ENTITY_TYPES:
            raise ValueError(f"unknown entity type {self.type!r}; expected one of {ENTITY_TYPES}")


@dataclass
class EntityScore:
    by_type: dict[str, tuple[int, int]]

    @property
    def overall(self) -> float:
        hit = sum(h for h, _ in self.by_type.values())
        tot = sum(t for _, t in self.by_type.values())
        return hit / tot if tot else 0.0

    def render(self) -> str:
        lines = [f"entity accuracy (overall): {self.overall * 100:5.1f}%"]
        for t in ENTITY_TYPES:
            if t in self.by_type:
                h, n = self.by_type[t]
                lines.append(f"  {t:<9} {h:>3}/{n:<3} {h / n * 100:5.1f}%")
        return "\n".join(lines)


def _present(surface: str, hypothesis: str) -> bool:
    """Whitespace-insensitive, case-insensitive containment on the normalised forms.

    Punctuation is kept -- "4:30" and "430" are different answers to a user, so the
    punctuated normalisation is the right comparison here even though WER also reports a
    stripped variant.
    """
    hay = normalise_for_wer(hypothesis, strip_punctuation=False)
    needle = normalise_for_wer(surface, strip_punctuation=False)
    if not needle:
        return False
    # Collapse spaces on both sides so "4 baje" matches "4  baje".
    return re.sub(r"\s+", " ", needle) in re.sub(r"\s+", " ", hay)


def score_entities(
    entity_lists: Sequence[Sequence[Entity]],
    hypotheses: Sequence[str],
) -> EntityScore:
    """Per-type hit counts over the corpus."""
    if len(entity_lists) != len(hypotheses):
        raise ValueError(f"{len(entity_lists)} entity lists vs {len(hypotheses)} hypotheses")

    tally: dict[str, list[int]] = {}
    for ents, hyp in zip(entity_lists, hypotheses):
        for e in ents:
            slot = tally.setdefault(e.type, [0, 0])
            slot[1] += 1
            slot[0] += _present(e.surface, hyp)
    return EntityScore({k: (v[0], v[1]) for k, v in tally.items()})
