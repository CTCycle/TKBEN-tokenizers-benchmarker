from __future__ import annotations

import pytest

from tests.unit.benchmark_metric_test_support import (
    CORE_TOKENIZER_BENCHMARK_METRIC_KEYS,
    build_benchmark_metric_value_map,
    run_deterministic_benchmark,
)


@pytest.fixture(scope="module")
def benchmark_metric_values() -> dict[str, object]:
    result = run_deterministic_benchmark()
    return build_benchmark_metric_value_map(result)


def test_tokenizer_benchmark_metrics_are_observed_and_non_synthetic(
    benchmark_metric_values: dict[str, object],
) -> None:
    for metric_key in sorted(CORE_TOKENIZER_BENCHMARK_METRIC_KEYS):
        assert metric_key in benchmark_metric_values, (
            f"Missing computed benchmark metric '{metric_key}'"
        )
    assert float(benchmark_metric_values["eff.encode_tokens_per_second_mean"]) > 0.0
    assert float(benchmark_metric_values["eff.encode_chars_per_second_mean"]) > 0.0
    assert float(benchmark_metric_values["eff.end_to_end_wall_time_seconds"]) > 0.0
    assert float(benchmark_metric_values["lat.encode_latency_p50_ms"]) >= 0.0
    assert float(benchmark_metric_values["lat.encode_latency_p95_ms"]) >= float(
        benchmark_metric_values["lat.encode_latency_p50_ms"]
    )
    assert float(benchmark_metric_values["lat.encode_latency_p99_ms"]) >= float(
        benchmark_metric_values["lat.encode_latency_p95_ms"]
    )
