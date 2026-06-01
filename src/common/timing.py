from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import mean
from typing import Iterator

import numpy as np


def now() -> float:
    return time.perf_counter()


@contextmanager
def elapsed() -> Iterator[callable[[], float]]:
    start = now()
    yield lambda: now() - start


def latency_summary_ms(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "query_latency_mean_ms": 0.0,
            "query_latency_p50_ms": 0.0,
            "query_latency_p95_ms": 0.0,
            "query_latency_max_ms": 0.0,
            "query_count": 0,
        }
    arr = np.array(values, dtype=float) * 1000.0
    return {
        "query_latency_mean_ms": float(mean(arr)),
        "query_latency_p50_ms": float(np.percentile(arr, 50)),
        "query_latency_p95_ms": float(np.percentile(arr, 95)),
        "query_latency_max_ms": float(np.max(arr)),
        "query_count": int(len(values)),
    }


@dataclass
class Timing:
    offline_index_time_seconds: float = 0.0
    offline_embedding_time_seconds: float = 0.0
    model_load_time_seconds: float = 0.0
    query_latencies_seconds: list[float] = field(default_factory=list)
    extra: dict[str, float | int | str | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int | str | bool]:
        data: dict[str, float | int | str | bool] = {
            "offline_index_time_seconds": self.offline_index_time_seconds,
            "offline_embedding_time_seconds": self.offline_embedding_time_seconds,
            "model_load_time_seconds": self.model_load_time_seconds,
        }
        data.update(latency_summary_ms(self.query_latencies_seconds))
        data.update(self.extra)
        return data
