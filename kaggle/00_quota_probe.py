"""Gate 0 -- does a T4x2 session bill wall-clock or GPU-hours?

    Accelerator: GPU T4 x2      Internet: not required      Quota: ~15-30 min

The 30 h/week pool is shared across P100 and T4x2, and nothing in Kaggle's documentation
says which way the meter charges. It moves the schedule ~2x and decides whether the
three-ablation block is affordable at all.

HOW TO RUN
  1. Read the GPU meter at kaggle.com/settings ("Kaggle GPU  HH:MM / 30 hrs").
     Screenshot it. The meter rounds and refreshes lazily -- you want the before value
     in writing, not in memory.
  2. Set BEFORE_METER below, then Save & Run All (Commit).
  3. Optionally start a second GPU notebook while this runs -- whether it launches, queues
     or is refused gives you the concurrency cap for free. If you do, set
     N_CONCURRENT_SESSIONS to match and keep the runtimes equal.
  4. When it finishes, read the meter again and run the decode cell at the bottom.
  5. Separately: run a CPU-only notebook for 15 min and check the GPU meter again. If it
     did not move, CPU sessions are free -- the assumption the whole workflow rests on.
"""

import subprocess
import time

# ----------------------------------------------------------------------- configure me
RUN_MINUTES = 15
N_CONCURRENT_SESSIONS = 1  # set to 2 if you launch a second T4x2 commit alongside
BEFORE_METER = "00:00"     # copied from kaggle.com/settings before starting
# --------------------------------------------------------------------------------------


def session_facts() -> None:
    """Everything about this session worth writing down while we have it."""
    print("=" * 68)
    print("SESSION FACTS")
    print("=" * 68)

    try:
        import torch
        print(f"torch                {torch.__version__}")
        print(f"cuda available       {torch.cuda.is_available()}")
        print(f"device_count         {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  gpu{i}              {p.name}, {p.total_memory / 1e9:.1f} GB")
    except Exception as e:  # noqa: BLE001 - diagnostic script, report and continue
        print(f"torch probe failed: {e}")

    # The ~20 GB figure quoted for /kaggle/working is the OUTPUT cap, not the scratch
    # disk. This is the only way to learn the real number.
    for cmd in (["df", "-h", "/kaggle/working"], ["free", "-g"], ["nproc"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
            print(f"\n$ {' '.join(cmd)}\n{out.rstrip()}")
        except Exception as e:  # noqa: BLE001
            print(f"\n$ {' '.join(cmd)} -> {e}")


def hold(minutes: int) -> None:
    """Occupy the GPU for `minutes`, printing a heartbeat so the cell is never silent.

    A trivial matmul keeps the device genuinely busy: if Kaggle ever meters utilisation
    rather than allocation, an idle sleep would read differently from real training.
    """
    print("\n" + "=" * 68)
    print(f"HOLDING FOR {minutes} MIN -- do not close the browser before the commit saves")
    print("=" * 68, flush=True)

    try:
        import torch
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        tensors = [torch.randn(2048, 2048, device=d) for d in devices]
    except Exception:  # noqa: BLE001
        devices, tensors = [], []

    start = time.time()
    for m in range(minutes):
        deadline = start + (m + 1) * 60
        while time.time() < deadline:
            for t in tensors:
                t @ t
            if not tensors:
                time.sleep(1)
        print(f"  minute {m + 1:>3}/{minutes}  elapsed {(time.time() - start) / 60:5.1f} min",
              flush=True)

    print(f"\ndone: {(time.time() - start) / 60:.2f} min of wall clock on "
          f"{len(devices)} GPU(s)")


def decode(after_meter: str) -> None:
    """Run this AFTER reading the meter again. Prints the verdict and the schedule."""
    import sys
    sys.path.insert(0, "/kaggle/working/whisper_distill/src")
    from whisper_distill.labeling.quota import decode_quota_probe

    def to_min(hhmm: str) -> float:
        h, m = hhmm.strip().split(":")
        return int(h) * 60 + int(m)

    delta = to_min(after_meter) - to_min(BEFORE_METER)
    verdict = decode_quota_probe(
        meter_delta_minutes=delta,
        n_concurrent_sessions=N_CONCURRENT_SESSIONS,
        minutes_per_session=RUN_MINUTES,
    )
    print(verdict.render())


if __name__ == "__main__":
    session_facts()
    hold(RUN_MINUTES)
    print(
        "\nNEXT: read the GPU meter again, then call\n"
        "    decode(after_meter='HH:MM')\n"
        "and record the verdict in docs/01-kaggle-execution-plan.md"
    )
