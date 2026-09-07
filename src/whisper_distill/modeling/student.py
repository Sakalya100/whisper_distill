"""Build the student in one call, in the order the surgery steps depend on each other."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from whisper_distill.config import DEFAULT, StudentConfig
from whisper_distill.modeling.surgery import (
    cut_decoder_layers,
    freeze_encoder,
    prune_vocabulary,
    select_vocabulary,
    shrink_input_window,
)

log = logging.getLogger(__name__)


def build_student(
    cfg: StudentConfig | None = None,
    *,
    token_counts: Mapping[int, int] | None = None,
    always_keep: Sequence[int] | None = None,
    use_fallback_init: bool = False,
):
    """Load the init checkpoint and apply the three cuts.

    Order matters. Decoder cut first (cheapest, and independent), then window shrink, then
    vocabulary prune last -- pruning rewrites `config.vocab_size` and re-ties weights, so
    anything that rebuilds modules should happen before it.

    `token_counts` comes from the pseudo-label corpus. Pass ``None`` to skip pruning, which
    is the right thing for the step-3 smoke test: prove the decoder cut trains before
    adding a second variable.

    Returns ``(model, old_to_new_token_ids | None)``.
    """
    from transformers import WhisperForConditionalGeneration

    cfg = cfg or DEFAULT.student
    model_id = cfg.init_fallback_id if use_fallback_init else cfg.init_model_id
    log.info("student init: %s", model_id)

    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model = cut_decoder_layers(model, cfg.n_decoder_layers)
    model = shrink_input_window(model, cfg.window_seconds)

    mapping = None
    if token_counts is not None:
        if always_keep is None:
            raise ValueError(
                "always_keep is required when pruning -- dropping BOS/EOS/pad or the "
                "language and task tokens yields a model that cannot start decoding"
            )
        keep = select_vocabulary(token_counts, cfg.target_vocab_size, always_keep)
        mapping = prune_vocabulary(model, keep)

    if cfg.freeze_encoder:
        model = freeze_encoder(model)

    return model, mapping
