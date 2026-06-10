from __future__ import annotations

from tests.unit.benchmark_metric_test_support import (
    BENCHMARK_METRIC_KEYS,
    DATASET_BENCHMARK_METRIC_KEYS,
    TOKENIZER_BENCHMARK_METRIC_KEYS,
)


###############################################################################
def test_benchmark_metric_partition_covers_entire_catalog() -> None:
    overlap = DATASET_BENCHMARK_METRIC_KEYS.intersection(
        TOKENIZER_BENCHMARK_METRIC_KEYS
    )
    assert not overlap, f"Metric partition overlap detected: {sorted(overlap)}"

    combined = DATASET_BENCHMARK_METRIC_KEYS.union(TOKENIZER_BENCHMARK_METRIC_KEYS)
    assert combined == BENCHMARK_METRIC_KEYS, (
        "Metric partition does not match benchmark catalog keys. "
        f"Missing={sorted(BENCHMARK_METRIC_KEYS - combined)} "
        f"Extra={sorted(combined - BENCHMARK_METRIC_KEYS)}"
    )


###############################################################################
def test_dataset_benchmark_expected_fixture_matches_partition() -> None:
    expected_keys = set(TOKENIZER_BENCHMARK_METRIC_KEYS).union(
        DATASET_BENCHMARK_METRIC_KEYS
    )
    assert expected_keys == BENCHMARK_METRIC_KEYS, (
        "Expected benchmark metric fixture is not aligned with catalog keys. "
        f"Missing={sorted(BENCHMARK_METRIC_KEYS - expected_keys)} "
        f"Extra={sorted(expected_keys - BENCHMARK_METRIC_KEYS)}"
    )
