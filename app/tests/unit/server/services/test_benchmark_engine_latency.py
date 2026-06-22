from __future__ import annotations

from server.domain.benchmark_observations import BatchObservation
from server.services.benchmark_engine import summarize_observations

###############################################################################
def test_latency_percentiles_come_from_samples() -> None:
    observations = [
        BatchObservation("t", 0, 0, 1, 10, 1, 0, 10_000_000, 1.0),
        BatchObservation("t", 0, 1, 1, 10, 1, 0, 20_000_000, 1.0),
        BatchObservation("t", 0, 2, 1, 10, 1, 0, 40_000_000, 1.0),
    ]
    summary = summarize_observations(observations)
    assert float(summary["latency_batch_p50_ms"]) == 20.0
    assert float(summary["latency_batch_p95_ms"]) >= 20.0
    assert float(summary["latency_batch_p99_ms"]) >= float(
        summary["latency_batch_p95_ms"]
    )
