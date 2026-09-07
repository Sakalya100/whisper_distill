"""Distillation objective, and the vocabulary-pruning trap it exists to prevent.

Run 1 is cross-entropy only -- shrink-and-fine-tune, a named variant in the
Distil-Whisper training docs. KL arrives later as a controlled ablation on a 50 h subset.
See docs/decisions/0003-ce-first-kl-as-ablation.md.

**The trap.** The student's vocabulary is pruned from ~51.8k to ~16k. A KL target computed
over the teacher's *full* vocabulary against the student's *pruned* vocabulary will train
without error and produce garbage: the two distributions are over different supports, and
the teacher's probability mass sitting on dropped tokens is never accounted for. The
teacher logits must be indexed down to the kept ids and **renormalised** before the KL.

Every function here asserts that alignment rather than documenting it, because the failure
mode is silent. Covered by tests/test_distill_loss.py (skipped without torch).

Status: written, **not yet executed** -- no torch in the local dev environment. First run
is Kaggle step 3.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

NEG_INF = float("-inf")


def build_token_lookup(old_to_new: Mapping[int, int], full_vocab_size: int):
    """Dense ``old_id -> new_id`` lookup, ``-1`` where the token was pruned away.

    A dict lookup per token per timestep is far too slow in the loss; this is the tensor
    form, built once at setup and reused.
    """
    import torch

    lut = torch.full((full_vocab_size,), -1, dtype=torch.long)
    for old, new in old_to_new.items():
        if not 0 <= old < full_vocab_size:
            raise ValueError(f"old id {old} outside teacher vocab of {full_vocab_size}")
        lut[old] = new
    return lut


def cross_entropy_loss(student_logits, labels, *, label_smoothing: float = 0.0):
    """Token-level CE over the student's own vocabulary. ``-100`` positions are ignored.

    `labels` must already be remapped to student ids. Passing original teacher ids here is
    the other half of the pruning trap -- it indexes the wrong rows and still converges to
    something, so it is checked, not assumed.
    """
    import torch.nn.functional as F

    vocab = student_logits.size(-1)
    valid = labels[labels != -100]
    if valid.numel() and int(valid.max()) >= vocab:
        raise ValueError(
            f"label id {int(valid.max())} >= student vocab {vocab}. Labels look un-remapped "
            "-- apply the old_to_new mapping from prune_vocabulary() first."
        )
    return F.cross_entropy(
        student_logits.reshape(-1, vocab).float(),
        labels.reshape(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )


def restricted_kl(
    student_logits,
    teacher_logits,
    keep_index,
    *,
    temperature: float = 2.0,
    mask=None,
):
    """KL(teacher || student) with the teacher restricted to the student's vocabulary.

    Args:
        student_logits: ``(B, T, V_student)``.
        teacher_logits: ``(B, T, V_teacher)`` -- the full-width teacher output.
        keep_index: ``LongTensor(V_student,)`` of teacher ids the student kept, ascending.
            This is ``sorted(old_to_new)``; index i must be the teacher id of student id i.
        mask: ``(B, T)`` bool/float, ``True`` on positions that count. Defaults to all.

    Renormalising after the index_select is the whole point: `log_softmax` over the
    restricted slice redistributes the mass that sat on dropped tokens, instead of leaving
    a distribution that does not sum to 1.
    """
    import torch
    import torch.nn.functional as F

    v_student = student_logits.size(-1)
    if keep_index.numel() != v_student:
        raise ValueError(
            f"keep_index has {keep_index.numel()} entries but student vocab is {v_student}; "
            "the KL target and the student head must be over the same token set"
        )
    if teacher_logits.shape[:-1] != student_logits.shape[:-1]:
        raise ValueError(
            f"teacher {tuple(teacher_logits.shape)} and student {tuple(student_logits.shape)} "
            "disagree on batch/time"
        )
    if int(keep_index.max()) >= teacher_logits.size(-1):
        raise ValueError("keep_index references a token id beyond the teacher's vocabulary")

    t_slice = teacher_logits.index_select(-1, keep_index.to(teacher_logits.device))
    t_logp = F.log_softmax(t_slice.float() / temperature, dim=-1)
    s_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)

    per_token = F.kl_div(s_logp, t_logp, log_target=True, reduction="none").sum(-1)

    if mask is None:
        mask = torch.ones_like(per_token)
    mask = mask.to(per_token.dtype)
    denom = mask.sum().clamp_min(1.0)
    # T^2 keeps the gradient magnitude comparable to the CE term as temperature varies.
    return (per_token * mask).sum() / denom * (temperature ** 2)


def kl_from_topk(
    student_logits,
    topk_values,
    topk_indices,
    token_lut,
    *,
    temperature: float = 2.0,
    mask=None,
):
    """KL against precomputed top-k teacher logits, mapped into student vocabulary space.

    Storing full 51.8k-wide teacher logits for 50 h of audio is not affordable on Kaggle
    disk, so the KL ablation caches top-k instead. That makes the vocabulary alignment
    trickier, not simpler: top-k ids are *teacher* ids, some of which the student pruned
    away, and those have to be dropped before renormalising.

    Args:
        topk_values: ``(B, T, K)`` teacher logits.
        topk_indices: ``(B, T, K)`` teacher token ids.
        token_lut: from `build_token_lookup` -- ``-1`` marks a pruned token.

    Positions whose entire top-k fell outside the student vocabulary carry no usable
    target and are dropped from the mask rather than producing NaN.
    """
    import torch
    import torch.nn.functional as F

    b, t, v_student = student_logits.shape
    if topk_values.shape != topk_indices.shape:
        raise ValueError("topk_values and topk_indices must have the same shape")
    if topk_values.shape[:2] != (b, t):
        raise ValueError(
            f"top-k tensor {tuple(topk_values.shape)} disagrees with student ({b}, {t}, ...)"
        )

    mapped = token_lut.to(topk_indices.device)[topk_indices]  # (B, T, K), -1 where pruned
    keepable = mapped >= 0

    dense = student_logits.new_full((b, t, v_student), NEG_INF)
    safe = mapped.clamp_min(0)
    src = topk_values.float().masked_fill(~keepable, NEG_INF)
    dense.scatter_(-1, safe, src)

    # A row with no surviving top-k entry would log_softmax to NaN. Drop it instead.
    has_target = keepable.any(-1)
    dense = torch.where(has_target.unsqueeze(-1), dense, dense.new_zeros(1))

    t_logp = F.log_softmax(dense / temperature, dim=-1)
    s_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    per_token = F.kl_div(s_logp, t_logp, log_target=True, reduction="none").sum(-1)

    eff = has_target if mask is None else (has_target & mask.bool())
    eff = eff.to(per_token.dtype)
    denom = eff.sum().clamp_min(1.0)
    return (per_token * eff).sum() / denom * (temperature ** 2)


class DistillLoss:
    """Weighted CE + optional restricted KL.

    Instantiate once at setup so the keep-index tensor and lookup table are built a single
    time, then call per step.
    """

    def __init__(
        self,
        *,
        keep_ids: Sequence[int] | None = None,
        teacher_vocab_size: int | None = None,
        use_kl: bool = False,
        ce_weight: float = 1.0,
        kl_weight: float = 0.8,
        temperature: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        import torch

        self.use_kl = use_kl
        self.ce_weight = ce_weight
        self.kl_weight = kl_weight
        self.temperature = temperature
        self.label_smoothing = label_smoothing

        if use_kl and keep_ids is None:
            raise ValueError(
                "KL requires keep_ids so the teacher can be restricted to the student's "
                "vocabulary. Without it the objective is silently over mismatched supports."
            )

        self.keep_index = None
        self.token_lut = None
        if keep_ids is not None:
            keep = list(keep_ids)
            if keep != sorted(keep):
                raise ValueError("keep_ids must be ascending")
            self.keep_index = torch.tensor(keep, dtype=torch.long)
            if teacher_vocab_size is not None:
                self.token_lut = build_token_lookup(
                    {old: new for new, old in enumerate(keep)}, teacher_vocab_size
                )

    def __call__(
        self,
        student_logits,
        labels,
        *,
        teacher_logits=None,
        teacher_topk=None,
    ) -> dict:
        """Returns ``{"loss", "ce", "kl"}``; `kl` is ``None`` when the term is off."""
        ce = cross_entropy_loss(
            student_logits, labels, label_smoothing=self.label_smoothing
        )
        out = {"ce": ce, "kl": None, "loss": self.ce_weight * ce}
        if not self.use_kl:
            return out

        mask = labels != -100
        if teacher_logits is not None:
            kl = restricted_kl(
                student_logits, teacher_logits, self.keep_index,
                temperature=self.temperature, mask=mask,
            )
        elif teacher_topk is not None:
            if self.token_lut is None:
                raise ValueError("teacher_vocab_size is required to use cached top-k logits")
            values, indices = teacher_topk
            kl = kl_from_topk(
                student_logits, values, indices, self.token_lut,
                temperature=self.temperature, mask=mask,
            )
        else:
            raise ValueError("use_kl=True but neither teacher_logits nor teacher_topk given")

        out["kl"] = kl
        out["loss"] = self.ce_weight * ce + self.kl_weight * kl
        return out
