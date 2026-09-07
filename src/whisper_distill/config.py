"""Single source of truth for the numbers that appear in the plan.

Anything referenced in docs/ as a concrete figure should be readable from here, so the
code and the plan cannot drift apart silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Audio / feature geometry
#
# Whisper runs at exactly 100 mel frames per second (hop 160 at 16 kHz). The original spec
# said "80 x 500" for a 10 s window; 500 is the POST-CONVOLUTION encoder length, one stage
# later than the cache. See docs/decisions/0002-mel-config-and-cache-layout.md.
# --------------------------------------------------------------------------------------

SAMPLE_RATE = 16_000
HOP_LENGTH = 160
FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # 100

WHISPER_NATIVE_WINDOW_S = 30
WHISPER_NATIVE_FRAMES = WHISPER_NATIVE_WINDOW_S * FRAMES_PER_SECOND  # 3000

#: Encoder convolution stack downsamples the frame axis by 2.
ENCODER_STRIDE = 2


@dataclass(frozen=True)
class AudioConfig:
    """Feature geometry for one cached clip."""

    sample_rate: int = SAMPLE_RATE
    hop_length: int = HOP_LENGTH
    n_mels: int = 80  # 80 for whisper small/medium; large-v3 is 128
    window_seconds: float = 10.0

    @property
    def n_frames(self) -> int:
        """Mel frames in the student window. 10 s -> 1000."""
        return int(round(self.window_seconds * FRAMES_PER_SECOND))

    @property
    def n_encoder_positions(self) -> int:
        """Encoder positions after the conv stack. 1000 -> 500."""
        return self.n_frames // ENCODER_STRIDE

    @property
    def bytes_per_clip_fp16(self) -> int:
        return self.n_mels * self.n_frames * 2

    def cache_bytes(self, n_clips: int) -> int:
        return self.bytes_per_clip_fp16 * n_clips

    def assert_teacher_compatible(self, teacher_n_mels: int) -> None:
        """The single-pass feature-extraction win requires matching mel counts.

        A 128-mel teacher (large-v3, whisper-large-v3-vaani-hindi) forfeits it and needs
        its own feature pass. Fail loudly rather than silently producing garbage features.
        """
        if teacher_n_mels != self.n_mels:
            raise ValueError(
                f"Teacher uses {teacher_n_mels} mel bins, student cache uses {self.n_mels}. "
                "One feature pass cannot serve both -- either pick an 80-mel teacher or run "
                "a separate teacher feature pass. See docs/decisions/0001-teacher-checkpoint.md"
            )


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TeacherConfig:
    """Teacher checkpoint. See docs/decisions/0001-teacher-checkpoint.md for why this is
    not IndicWhisper: Vistaar ships a zip from object storage, not an HF repo."""

    model_id: str = "vasista22/whisper-hindi-medium"
    n_mels: int = 80
    language: str = "hi"
    task: str = "transcribe"

    #: Greedy only. Beam search spikes VRAM and OOMs on a T4 where greedy fits.
    num_beams: int = 1

    #: CTranslate2 compute type for the bulk labelling pass. int8_float16 is documented as
    #: tuned for Ampere+; T4 is Turing, so benchmark against "float16" before the full run.
    ct2_compute_type: str = "float16"

    #: 128-mel fallbacks if the Gate 1 code-switch audit fails.
    fallbacks: tuple[str, ...] = (
        "openai/whisper-large-v3",
        "ARTPARK-IISc/whisper-large-v3-vaani-hindi",
    )


@dataclass(frozen=True)
class StudentConfig:
    """Student surgery targets. Param counts are from the plan's table."""

    init_model_id: str = "vasista22/whisper-hindi-small"

    #: If the Gate 1 audit shows the Hindi fine-tune lost English, fall back to this and
    #: absorb the extra epochs.
    init_fallback_id: str = "openai/whisper-small"

    n_decoder_layers: int = 4  # from 12
    target_vocab_size: int = 16_000  # from ~51.8k
    window_seconds: float = 10.0

    #: Frozen encoder: no gradients or optimiser state for ~88M params, which roughly
    #: doubles usable batch size on a T4 and preserves acoustic robustness. Encoder
    #: shrinking is an end-of-project ablation, not a first-run decision.
    freeze_encoder: bool = True


# --------------------------------------------------------------------------------------
# Distillation
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DistillConfig:
    """Run 1 is cross-entropy only -- shrink-and-fine-tune, a named Distil-Whisper variant.
    KL is added later on a 50 h subset as a controlled ablation.
    See docs/decisions/0003-ce-first-kl-as-ablation.md."""

    use_kl: bool = False
    ce_weight: float = 1.0
    kl_weight: float = 0.8
    temperature: float = 2.0

    #: Top-k teacher logits kept when precomputing for the KL ablation. Storing full
    #: 51.8k-wide logits for 50 h of audio is not affordable on Kaggle disk.
    logit_top_k: int = 64


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DataSource:
    hf_id: str
    config: str | None
    target_hours: float
    gated: bool = True
    license: str = "cc-by-4.0"


@dataclass(frozen=True)
class DataConfig:
    sources: tuple[DataSource, ...] = (
        # Vaani's headline "2043 hours" is the transcribed subset, not the corpus. The
        # Hindi config holds 963 h; all 59 configs total 234 GB -- stream with early stop.
        DataSource("ARTPARK-IISc/Vaani-transcription-part", "Hindi", 120.0),
        DataSource("ai4bharat/IndicVoices", "hindi", 50.0),
    )
    scraped_hinglish_hours: float = 30.0

    #: Drop a pseudo-label whose WER against an existing human reference exceeds this.
    #: The Distil-Whisper WER filter, which the paper shows matters a lot.
    max_teacher_wer: float = 0.20

    #: Proxy filter for scraped audio, which has no reference. Tune on labelled data first.
    max_mean_token_entropy: float = 2.5

    min_clip_seconds: float = 1.0
    max_clip_seconds: float = 10.0

    #: Kaggle datasets cap at 50 top-level files, so 72k loose .npy files is not an option.
    shard_target_bytes: int = 4 * 1024**3


@dataclass(frozen=True)
class KaggleConfig:
    """Platform limits. UNVERIFIED entries are settled by kaggle/00_quota_probe.py."""

    # --- confirmed on the account, 2026-09-08
    weekly_gpu_hours: int = 30  # one pool shared across P100 and T4x2
    weekly_tpu_hours: int = 20  # unusable: transformers Trainer crashes TPU sessions
    session_hours: int = 12
    working_dir_cap_bytes: int = 20 * 1024**3

    #: 200 GiB. The UI reports this as "214.75 GB" because it converts to decimal GB --
    #: the same number, so do not budget 200 decimal GB and think there is headroom.
    private_dataset_cap_bytes: int = 200 * 1024**3
    dataset_max_top_level_files: int = 50

    #: Confirmed: two T4x2 committed runs launched and ran concurrently. Note that
    #: concurrent sessions burn quota in parallel -- useful for the ablation block only if
    #: billing turns out to be per-session wall-clock.
    max_concurrent_gpu_sessions: int = 2

    # --- still unverified; settled by kaggle/00_quota_probe.py
    #: Does a T4x2 session bill wall-clock or GPU-hours? ~2x schedule impact.
    t4x2_bills_wall_clock: bool | None = None

    #: Assumed True; the whole "prep on CPU" strategy rests on it.
    cpu_sessions_are_free: bool | None = None


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    data: DataConfig = field(default_factory=DataConfig)
    kaggle: KaggleConfig = field(default_factory=KaggleConfig)


DEFAULT = Config()
