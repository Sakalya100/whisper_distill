"""Student surgery: decoder cut, input-window shrink, vocabulary prune.

Three independent operations on a `WhisperForConditionalGeneration`, applied in that
order. Each is reversible only by reloading the checkpoint, so each one asserts its own
postcondition rather than trusting the caller.

    whisper-small Hindi baseline          ~244M
    decoder 12 -> 4 layers                ~160M
    vocabulary ~51.8k -> ~16k             ~132M
    10 s window (positional embeds sliced) ~132M, ~3x less encoder compute

Status: written against the transformers Whisper module layout; **not yet executed** --
there is no torch in the local dev environment. First run is Kaggle step 3
(`kaggle/03_smoke_train_gpu.py`), which is exactly the 1-GPU-hour smoke test that exists
to catch this. `maximally_spaced_indices` and `select_vocabulary` are pure Python and are
covered by tests/test_surgery.py.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence

from whisper_distill.config import ENCODER_STRIDE, FRAMES_PER_SECOND

log = logging.getLogger(__name__)


# ======================================================================================
# Pure helpers -- no torch, unit-tested
# ======================================================================================

def maximally_spaced_indices(n_from: int, n_to: int) -> list[int]:
    """Indices of `n_to` layers spread as evenly as possible across `n_from`, endpoints
    included.

    This is the Distil-Whisper shrink-and-fine-tune initialisation: copy maximally spaced
    layers from the teacher/parent rather than the first N. For ``n_to == 2`` it reduces to
    first-and-last, which is what Distil-Whisper's own 2-layer decoder uses.

    >>> maximally_spaced_indices(12, 4)
    [0, 4, 7, 11]
    >>> maximally_spaced_indices(12, 2)
    [0, 11]
    """
    if n_to < 1:
        raise ValueError(f"n_to must be >= 1, got {n_to}")
    if n_to > n_from:
        raise ValueError(f"cannot keep {n_to} of {n_from} layers")
    if n_to == 1:
        return [0]
    step = (n_from - 1) / (n_to - 1)
    idx = [round(i * step) for i in range(n_to)]
    # Rounding can collide on adjacent slots for near-full keeps; nudge to stay strictly
    # increasing so we never copy the same parent layer twice.
    for i in range(1, len(idx)):
        if idx[i] <= idx[i - 1]:
            idx[i] = idx[i - 1] + 1
    if idx[-1] > n_from - 1:
        raise ValueError(f"index overflow selecting {n_to} of {n_from}")
    return idx


def select_vocabulary(
    token_counts: Mapping[int, int],
    target_size: int,
    always_keep: Iterable[int],
) -> list[int]:
    """Choose which token ids the pruned student keeps.

    `token_counts` is the frequency of every token id across the pseudo-label corpus --
    build it in the CPU stage once the labels exist, never guess it. `always_keep` holds
    the ids the model cannot function without (BOS/EOS/pad, the language and task tokens,
    ``<|notimestamps|>``); those are kept regardless of frequency and counted against the
    budget.

    Returns sorted ascending, so the mapping to new ids is order-preserving and a
    dropped-token check on the original ids stays monotonic.

    A token id absent from the returned list becomes unreachable for the student. Any
    pseudo-label containing one must be dropped or re-tokenised -- see
    `assert_labels_within_vocabulary`.
    """
    forced = set(always_keep)
    if target_size < len(forced):
        raise ValueError(
            f"target_size {target_size} is smaller than the {len(forced)} mandatory tokens"
        )
    budget = target_size - len(forced)
    ranked = sorted(
        (tid for tid in token_counts if tid not in forced),
        key=lambda tid: (-token_counts[tid], tid),
    )
    return sorted(forced | set(ranked[:budget]))


def assert_labels_within_vocabulary(
    label_ids: Sequence[int],
    old_to_new: Mapping[int, int],
    ignore: Iterable[int] = (-100,),
) -> None:
    """Fail loudly if a pseudo-label references a token the student can no longer emit.

    Silently clamping or dropping these is how vocabulary pruning produces a model that
    trains cleanly and outputs nonsense.
    """
    ignored = set(ignore)
    missing = {t for t in label_ids if t not in ignored and t not in old_to_new}
    if missing:
        raise ValueError(
            f"{len(missing)} label token(s) fall outside the pruned vocabulary, "
            f"e.g. {sorted(missing)[:8]}. Re-filter the corpus or widen target_vocab_size."
        )


# ======================================================================================
# Model surgery -- imports torch lazily so the pure helpers stay importable on CPU-only
# ======================================================================================

def cut_decoder_layers(model, n_keep: int):
    """Keep `n_keep` maximally spaced decoder layers in place.

    Mutates and returns `model`. Encoder is untouched: on free-tier compute the encoder
    stays whole and frozen (no gradients or optimiser state for ~88M params roughly
    doubles usable batch size on a T4, and the full encoder keeps acoustic robustness).
    """
    import torch.nn as nn

    layers = model.model.decoder.layers
    keep = maximally_spaced_indices(len(layers), n_keep)
    log.info("decoder %d -> %d layers, keeping parent indices %s", len(layers), n_keep, keep)

    model.model.decoder.layers = nn.ModuleList([layers[i] for i in keep])
    model.config.decoder_layers = n_keep
    if hasattr(model.generation_config, "decoder_layers"):
        model.generation_config.decoder_layers = n_keep

    assert len(model.model.decoder.layers) == n_keep
    return model


def shrink_input_window(model, window_seconds: float):
    """Slice the encoder's positional embeddings down to a shorter input window.

    Whisper was only ever trained on 30-second zero-padded input, so this is
    architecturally clean but off-distribution -- expect degradation that fine-tuning has
    to recover. That is precisely why window length is one of the three ablations rather
    than a free win.

    The encoder's `embed_positions` is a fixed sinusoidal table, so slicing a prefix keeps
    every retained position's encoding exactly as the parent had it.
    """
    import torch
    import torch.nn as nn

    n_frames = int(round(window_seconds * FRAMES_PER_SECOND))
    n_positions = n_frames // ENCODER_STRIDE

    embed = model.model.encoder.embed_positions
    old_positions, d_model = embed.weight.shape
    if n_positions > old_positions:
        raise ValueError(
            f"requested {n_positions} encoder positions but parent has {old_positions}; "
            "cannot extend a sinusoidal table by slicing"
        )

    sliced = nn.Embedding(n_positions, d_model)
    with torch.no_grad():
        sliced.weight.copy_(embed.weight[:n_positions])
    sliced.requires_grad_(False)  # stays fixed, as in the parent
    model.model.encoder.embed_positions = sliced

    model.config.max_source_positions = n_positions
    log.info(
        "input window -> %.1fs (%d mel frames, %d encoder positions, was %d)",
        window_seconds, n_frames, n_positions, old_positions,
    )

    assert model.model.encoder.embed_positions.weight.shape[0] == n_positions
    return model


def prune_vocabulary(model, keep_ids: Sequence[int]) -> dict[int, int]:
    """Shrink the decoder embedding and output projection to `keep_ids`.

    Returns the ``old_id -> new_id`` mapping, which the caller **must** use to remap every
    pseudo-label and to index teacher logits before computing KL. Both matter:

    * Labels still holding original ids will index the wrong rows.
    * A KL target over the teacher's full vocabulary against a pruned student vocabulary
      trains fine and produces garbage. See `training.distill_loss.restricted_kl`.
    """
    import torch
    import torch.nn as nn

    keep = list(keep_ids)
    if len(set(keep)) != len(keep):
        raise ValueError("keep_ids contains duplicates")
    if keep != sorted(keep):
        raise ValueError("keep_ids must be sorted ascending so new ids stay order-preserving")

    old_embed = model.model.decoder.embed_tokens
    old_vocab, d_model = old_embed.weight.shape
    if keep[-1] >= old_vocab:
        raise ValueError(f"keep_ids references id {keep[-1]} beyond vocab size {old_vocab}")

    index = torch.tensor(keep, dtype=torch.long, device=old_embed.weight.device)

    new_embed = nn.Embedding(len(keep), d_model, padding_idx=old_embed.padding_idx)
    with torch.no_grad():
        new_embed.weight.copy_(old_embed.weight.index_select(0, index))
    model.model.decoder.embed_tokens = new_embed

    # proj_out is weight-tied to embed_tokens in Whisper. Rebuild it and re-tie so the
    # tie survives a save/load round trip rather than silently detaching.
    new_proj = nn.Linear(d_model, len(keep), bias=False)
    with torch.no_grad():
        new_proj.weight.copy_(new_embed.weight)
    model.proj_out = new_proj
    model.config.vocab_size = len(keep)
    model.tie_weights()

    log.info("vocabulary %d -> %d tokens", old_vocab, len(keep))
    assert model.proj_out.weight.shape[0] == len(keep)
    assert model.model.decoder.embed_tokens.weight.shape[0] == len(keep)

    return {old: new for new, old in enumerate(keep)}


def freeze_encoder(model):
    """Freeze the encoder and confirm nothing in it still wants gradients."""
    model.model.encoder.requires_grad_(False)
    # Whisper's own helper also handles the gradient checkpointing interaction.
    if hasattr(model, "freeze_encoder"):
        model.freeze_encoder()
    live = [n for n, p in model.model.encoder.named_parameters() if p.requires_grad]
    if live:
        raise RuntimeError(f"encoder params still trainable after freeze: {live[:4]}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("frozen encoder: %.1fM trainable of %.1fM total", trainable / 1e6, total / 1e6)
    return model
