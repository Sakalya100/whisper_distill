"""Shard writer for the mel + token cache.

Kaggle datasets cap at **50 top-level files**, so 72,000 loose .npy files is not an option.
Layout, ~4 GB per shard:

    shard-0000.mels.npy     (N, n_mels, n_frames) float16
    shard-0000.tokens.npy   (N, max_len) int32, right-padded with -100
    index.json              per-clip metadata, including which shard/row

Read back memory-mapped from /kaggle/input, which turns the dataloader from a CPU
bottleneck into a page-cache read.

numpy only -- runs and is tested in a CPU session.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from whisper_distill.config import DEFAULT

log = logging.getLogger(__name__)

PAD_TOKEN = -100
MEL_DTYPE = np.float16
TOKEN_DTYPE = np.int32

#: Whisper's log-mel is normalised as ``(log10(mag) + 4) / 4`` after the floor
#: ``max(log_spec, log_spec.max() - 8)``. In that normalised space the floor therefore sits
#: exactly ``8/4 = 2.0`` below the clip's own maximum -- for any input, no approximation.
#: So the region Whisper pads is a negative constant (typically ~-0.5 to -1.0), NOT zero.
#: Zero-padding here would feed a FROZEN encoder a value it never saw in training, and a
#: frozen encoder cannot adapt to the new convention.
WHISPER_FLOOR_OFFSET = 2.0


def whisper_floor(mel: np.ndarray) -> float:
    """The value Whisper's own feature extractor puts in padded frames, for this clip."""
    return float(np.max(mel)) - WHISPER_FLOOR_OFFSET


@dataclass
class ClipRecord:
    """One cached clip. Everything a filter or a dataloader needs, without touching audio."""

    clip_id: str
    shard: int
    row: int
    n_frames: int
    n_tokens: int
    source: str = ""
    duration_s: float = 0.0
    text: str = ""
    #: Frames of real audio in the window; the rest is Whisper-floor padding.
    #: Teacher WER against a human reference, where one exists. None for scraped audio.
    teacher_wer: float | None = None
    #: Mean per-token entropy of the teacher -- the proxy filter for unreferenced audio.
    mean_entropy: float | None = None


@dataclass
class ShardIndex:
    n_mels: int
    n_frames: int
    max_tokens: int
    records: list[ClipRecord] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "n_mels": self.n_mels,
                    "n_frames": self.n_frames,
                    "max_tokens": self.max_tokens,
                    "records": [asdict(r) for r in self.records],
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> ShardIndex:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            n_mels=raw["n_mels"],
            n_frames=raw["n_frames"],
            max_tokens=raw["max_tokens"],
            records=[ClipRecord(**r) for r in raw["records"]],
        )

    def filtered(
        self,
        *,
        max_teacher_wer: float | None = None,
        max_mean_entropy: float | None = None,
    ) -> list[ClipRecord]:
        """Apply the Distil-Whisper WER filter and the entropy proxy.

        A clip with no reference is judged on entropy; a clip with a reference is judged on
        WER and keeps its entropy unexamined. A clip with neither passes -- surfacing that
        as a warning rather than silently keeping or dropping it.
        """
        kept, unjudged = [], 0
        for r in self.records:
            if r.teacher_wer is not None and max_teacher_wer is not None:
                if r.teacher_wer > max_teacher_wer:
                    continue
            elif r.mean_entropy is not None and max_mean_entropy is not None:
                if r.mean_entropy > max_mean_entropy:
                    continue
            else:
                unjudged += 1
            kept.append(r)
        if unjudged:
            log.warning(
                "%d of %d clips passed unfiltered (no reference and no entropy recorded)",
                unjudged, len(self.records),
            )
        return kept


class ShardWriter:
    """Append clips, rolling to a new shard once the byte target is hit.

    Writes are buffered per shard and flushed on roll, so peak memory is one shard, not
    the whole corpus. Use as a context manager so the final partial shard is always
    flushed -- a Kaggle session that dies mid-write otherwise loses the tail silently.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        n_mels: int | None = None,
        n_frames: int | None = None,
        max_tokens: int = 448,
        shard_target_bytes: int | None = None,
    ):
        audio = DEFAULT.audio
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.n_mels = n_mels or audio.n_mels
        self.n_frames = n_frames or audio.n_frames
        self.max_tokens = max_tokens
        self.target = shard_target_bytes or DEFAULT.data.shard_target_bytes

        self.index = ShardIndex(self.n_mels, self.n_frames, self.max_tokens)
        self._shard = 0
        self._mels: list[np.ndarray] = []
        self._tokens: list[np.ndarray] = []
        self._bytes = 0

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def add(
        self,
        clip_id: str,
        mel: np.ndarray,
        token_ids: list[int],
        *,
        speech_frames: int | None = None,
        pad_value: float | None = None,
        **meta,
    ) -> ClipRecord:
        """Add one clip.

        **Prefer passing a full-width mel.** Slice the teacher's own feature tensor --
        ``feats[i, :, :n_frames]`` -- so the padded columns carry Whisper's floor exactly as
        the encoder expects. That is the whole point of the prefix property, and it also
        avoids re-deriving a frame count from a rounded duration.

        A shorter mel is accepted and right-padded with `whisper_floor(mel)`, which
        reproduces the same constant. It is **not** zero-padded: Whisper's padded region is
        a negative constant, and a frozen encoder cannot adapt to a different convention.

        `speech_frames` records how much of the window is real audio, for metadata only.
        """
        if mel.ndim != 2 or mel.shape[0] != self.n_mels:
            raise ValueError(
                f"mel must be ({self.n_mels}, <={self.n_frames}), got {mel.shape}"
            )
        if mel.shape[1] > self.n_frames:
            raise ValueError(
                f"mel has {mel.shape[1]} frames, cache window is {self.n_frames}. "
                "Segment to <=10 s before labelling -- see data/segment.py"
            )
        if len(token_ids) > self.max_tokens:
            raise ValueError(f"{len(token_ids)} tokens exceeds max_tokens={self.max_tokens}")

        if mel.shape[1] == self.n_frames:
            padded_mel = mel.astype(MEL_DTYPE)
        else:
            fill = whisper_floor(mel) if pad_value is None else pad_value
            padded_mel = np.full((self.n_mels, self.n_frames), fill, dtype=MEL_DTYPE)
            padded_mel[:, : mel.shape[1]] = mel.astype(MEL_DTYPE)

        padded_tok = np.full(self.max_tokens, PAD_TOKEN, dtype=TOKEN_DTYPE)
        padded_tok[: len(token_ids)] = np.asarray(token_ids, dtype=TOKEN_DTYPE)

        rec = ClipRecord(
            clip_id=clip_id,
            shard=self._shard,
            row=len(self._mels),
            n_frames=int(speech_frames if speech_frames is not None else mel.shape[1]),
            n_tokens=len(token_ids),
            **meta,
        )
        self._mels.append(padded_mel)
        self._tokens.append(padded_tok)
        self._bytes += padded_mel.nbytes + padded_tok.nbytes
        self.index.records.append(rec)

        if self._bytes >= self.target:
            self._flush()
        return rec

    def _flush(self) -> None:
        if not self._mels:
            return
        stem = self.out_dir / f"shard-{self._shard:04d}"
        np.save(f"{stem}.mels.npy", np.stack(self._mels))
        np.save(f"{stem}.tokens.npy", np.stack(self._tokens))
        log.info("wrote %s (%d clips, %.2f GB)", stem.name, len(self._mels), self._bytes / 1e9)
        self._shard += 1
        self._mels, self._tokens, self._bytes = [], [], 0

    def close(self) -> None:
        self._flush()
        self.index.save(self.out_dir / "index.json")
        n_files = self._shard * 2 + 1
        cap = DEFAULT.kaggle.dataset_max_top_level_files
        if n_files > cap:
            log.warning(
                "%d top-level files exceeds Kaggle's limit of %d -- raise shard_target_bytes",
                n_files, cap,
            )
        log.info("closed: %d clips across %d shards (%d files)",
                 len(self.index.records), self._shard, n_files)


def open_shards(cache_dir: str | Path) -> tuple[ShardIndex, dict[int, tuple]]:
    """Memory-map every shard. Returns ``(index, {shard: (mels, tokens)})``.

    ``mmap_mode="r"`` is the point of the whole layout -- pages are faulted in on demand
    from the read-only /kaggle/input mount instead of being read through Python.
    """
    cache_dir = Path(cache_dir)
    index = ShardIndex.load(cache_dir / "index.json")
    shards: dict[int, tuple] = {}
    for mel_path in sorted(cache_dir.glob("shard-*.mels.npy")):
        n = int(mel_path.name.split("-")[1].split(".")[0])
        shards[n] = (
            np.load(mel_path, mmap_mode="r"),
            np.load(cache_dir / f"shard-{n:04d}.tokens.npy", mmap_mode="r"),
        )
    if not shards:
        raise FileNotFoundError(f"no shard-*.mels.npy under {cache_dir}")
    return index, shards
