"""Safe iteration over a Hugging Face streaming dataset.

Abandoning an `IterableDataset` mid-download leaves the hub client's HTTP session and
torchcodec's FFmpeg threads running while the interpreter starts finalizing. What comes out
the other side is:

    [Errno 9] Bad file descriptor  ... Retrying in 1s [Retry 1/5].
    Fatal Python error: PyGILState_Release: auto-releasing thread-state,
                        but no thread-state for this thread
    Python runtime state: finalizing

The retry thread reaches for a descriptor the shutdown already closed, and a torchcodec
C++ thread calls into a GIL state that no longer exists.

It is harmless when it happens *after* the work is done and flushed -- but on Kaggle a
fatal Python error can mark the commit as failed, which blocks "Create Dataset from
Output". That turns a cosmetic race into a lost session, so it is worth closing properly.

`take` drops each row as soon as it is consumed, releases the iterator explicitly, forces a
collection, and gives the background threads a moment to notice before returning.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Iterator
from typing import Any

log = logging.getLogger(__name__)

#: Seconds to let hub retry threads and FFmpeg workers wind down before returning.
#: Empirically a second or two is enough; the cost is trivial against a 30-minute run.
SETTLE_SECONDS = 2.0


def take(dataset: Any, limit: int | None = None, *, settle: float = SETTLE_SECONDS) -> Iterator:
    """Yield up to `limit` rows, then tear the stream down cleanly.

    Use this instead of ``for row in dataset`` plus ``break``. The caller must not retain
    the yielded row (or its audio field) after the next iteration -- torchcodec decoders
    hold native resources, and keeping them alive is what puts threads in flight at exit.
    """
    it = iter(dataset)
    n = 0
    try:
        while limit is None or n < limit:
            try:
                row = next(it)
            except StopIteration:
                break
            n += 1
            yield row
            del row  # release the AudioDecoder before pulling the next shard
    finally:
        del it
        gc.collect()
        if settle > 0:
            time.sleep(settle)
        log.info("stream closed cleanly after %d rows", n)
