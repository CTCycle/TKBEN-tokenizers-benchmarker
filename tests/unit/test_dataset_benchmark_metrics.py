from __future__ import annotations

import pytest

from tests.unit.benchmark_metric_test_support import (
    BENCHMARK_METRIC_KEYS,
    DATASET_BENCHMARK_METRIC_KEYS,
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


def test_benchmark_metric_partition_covers_entire_catalog() -> None:
    overlap = DATASET_BENCHMARK_METRIC_KEYS.intersection(TOKENIZER_BENCHMARK_METRIC_KEYS)
    assert not overlap, f"Metric partition overlap detected: {sorted(overlap)}"

    combined = DATASET_BENCHMARK_METRIC_KEYS.union(TOKENIZER_BENCHMARK_METRIC_KEYS)
    assert combined == BENCHMARK_METRIC_KEYS, (
        "Metric partition does not match benchmark catalog keys. "
        f"Missing={sorted(BENCHMARK_METRIC_KEYS - combined)} "
        f"Extra={sorted(combined - BENCHMARK_METRIC_KEYS)}"
    )


def test_dataset_benchmark_expected_fixture_matches_partition() -> None:
    expected_keys = set(EXPECTED_BENCHMARK_METRIC_VALUES.keys())
    assert expected_keys == BENCHMARK_METRIC_KEYS, (
        "Expected benchmark metric fixture is not aligned with catalog keys. "
        f"Missing={sorted(BENCHMARK_METRIC_KEYS - expected_keys)} "
        f"Extra={sorted(expected_keys - BENCHMARK_METRIC_KEYS)}"
    )


@pytest.mark.parametrize("metric_key", sorted(DATASET_BENCHMARK_METRIC_KEYS))
def test_dataset_benchmark_metrics_are_deterministic_and_correct(
    benchmark_metric_values: dict[str, object],
    metric_key: str,
) -> None:
    assert metric_key in benchmark_metric_values, f"Missing computed benchmark metric '{metric_key}'"
    expected = EXPECTED_BENCHMARK_METRIC_VALUES[metric_key]
    actual = benchmark_metric_values[metric_key]
    assert_metric_value(actual, expected, metric_key=metric_key)
