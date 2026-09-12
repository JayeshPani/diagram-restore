"""Batch-one restoration latency (docs/04_EXPERIMENT_PROTOCOL.md §8).

Screening-pass timing: fewer repeated sessions than the full protocol calls for.
Report these numbers as measured screening evidence, not the confirmatory timing study.
"""

import time

import numpy as np
import torch


def measure_latency(
    restore, warmup: int = 20, repeats: int = 50, device: torch.device | None = None
) -> dict:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive")

    def sync() -> None:
        if device is not None and device.type == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        restore()
    samples = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        restore()
        sync()
        samples.append(time.perf_counter() - start)
    return {
        "warmup": warmup,
        "repeats": repeats,
        "median_seconds": float(np.median(samples)),
        "p95_seconds": float(np.quantile(samples, 0.95)),
        "mean_seconds": float(np.mean(samples)),
    }
