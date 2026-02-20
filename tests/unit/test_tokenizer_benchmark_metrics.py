from __future__ import annotations

import pytest

from tests.unit.benchmark_metric_test_support import (
    EXPECTED_BENCHMARK_METRIC_VALUES,
    TOKENIZER_BENCHMARK_METRIC_KEYS,
    assert_metric_value,
    build_benchmark_metric_value_map,
    run_deterministic_benchmark,
)


@pytest.fixture(scope="module")
def benchmark_metric_values() -> dict[str, object]:
    result = run_deterministic_benchmark()
    return build_benchmark_metric_value_map(result)


@pytest.mark.parametrize("metric_key", sorted(TOKENIZER_BENCHMARK_METRIC_KEYS))
def test_tokenizer_benchmark_metrics_are_deterministic_and_correct(
    benchmark_metric_values: dict[str, object],
    metric_key: str,
) -> None:
    assert metric_key in benchmark_metric_values, f"Missing computed benchmark metric '{metric_key}'"
    expected = EXPECTED_BENCHMARK_METRIC_VALUES[metric_key]
    actual = benchmark_metric_values[metric_key]
    assert_metric_value(actual, expected, metric_key=metric_key)
