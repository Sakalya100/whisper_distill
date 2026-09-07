"""Pseudo-label quality filters.

Two filters, applied to different populations:

* **WER filter** where a human reference exists. Distil-Whisper's own ablations show this
  matters a lot -- it discards the cases where the teacher mis-transcribed or hallucinated,
  and the paper reports a large downstream WER improvement from it alone.
* **Entropy proxy** for scraped audio, which has no reference at all. Calibrate the
  threshold on referenced data first: pick the entropy cutoff that reproduces the WER
  filter's keep-rate, rather than guessing a number.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from whisper_distill.evaluation.metrics import wer


def passes_wer_filter(reference: str, pseudo_label: str, *, threshold: float = 0.20) -> bool:
    """Normalise both sides, compute WER, keep below `threshold`.

    Both strings go through the same normalisation, which is what makes this comparison
    fair: the human reference is unpunctuated and lower-cased while the pseudo-label
    carries punctuation and casing, so scoring the raw forms would reject good labels for
    having exactly the property we wanted from the teacher.
    """
    errors, length = wer(reference, pseudo_label)
    if length == 0:
        return False  # no reference content to verify against
    return (errors / length) <= threshold


def calibrate_entropy_threshold(
    entropies: Sequence[float],
    keeps: Sequence[bool],
) -> float:
    """Pick the entropy cutoff whose keep-rate matches the WER filter's on labelled data.

    `entropies` and `keeps` come from the referenced subset: the teacher's mean token
    entropy per clip, and whether `passes_wer_filter` kept it. The returned threshold is
    then applied to unreferenced scraped audio.
    """
    if len(entropies) != len(keeps):
        raise ValueError("entropies and keeps must align")
    if not entropies:
        raise ValueError("need at least one calibration clip")
    target_rate = sum(keeps) / len(keeps)
    ordered = sorted(entropies)
    cut = max(0, min(len(ordered) - 1, int(round(target_rate * len(ordered))) - 1))
    return ordered[cut]


def mean_token_entropy(token_logprobs: Sequence[Sequence[float]]) -> float:
    """Mean per-token predictive entropy, in nats, from per-step log-probabilities.

    `token_logprobs` is one sequence of log-probs per decoded step (the top-k tail is
    enough in practice). Renormalised per step, so a truncated top-k does not inflate the
    entropy simply by summing to less than 1.
    """
    if not token_logprobs:
        return 0.0
    total = 0.0
    steps = 0
    for step in token_logprobs:
        probs = [math.exp(lp) for lp in step]
        z = sum(probs)
        if z <= 0:
            continue
        h = -sum((p / z) * math.log(p / z) for p in probs if p > 0)
        total += h
        steps += 1
    return total / steps if steps else 0.0
