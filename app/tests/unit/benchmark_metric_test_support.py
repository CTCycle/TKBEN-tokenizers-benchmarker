from __future__ import annotations

from typing import Any

import pytest

from server.services.benchmarks import BENCHMARK_METRIC_CATALOG, BenchmarkService


BENCHMARK_METRIC_KEYS: set[str] = {
    metric["key"]
    for category in BENCHMARK_METRIC_CATALOG
    for metric in category.get("metrics", [])
    if isinstance(metric, dict) and isinstance(metric.get("key"), str)
}
CORE_BENCHMARK_METRIC_KEYS: set[str] = {
    metric["key"]
    for category in BENCHMARK_METRIC_CATALOG
    for metric in category.get("metrics", [])
    if (
        isinstance(metric, dict)
        and isinstance(metric.get("key"), str)
        and bool(metric.get("core"))
    )
}

EXPECTED_BENCHMARK_METRIC_VALUES: dict[str, Any] = {}

DATASET_BENCHMARK_METRIC_KEYS: set[str] = set()

TOKENIZER_BENCHMARK_METRIC_KEYS: set[str] = (
    BENCHMARK_METRIC_KEYS - DATASET_BENCHMARK_METRIC_KEYS
)
CORE_TOKENIZER_BENCHMARK_METRIC_KEYS: set[str] = (
    CORE_BENCHMARK_METRIC_KEYS - DATASET_BENCHMARK_METRIC_KEYS
)

###############################################################################
class DummyTokenizer:
    name_or_path = "dummy/tokenizer"

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {
            "alpha": 1,
            "beta": 2,
            "gamma": 3,
            "delta": 4,
            "##tail": 5,
        }
        self._id_to_token = {value: key for key, value in self._vocab.items()}

    # -------------------------------------------------------------------------
    def tokenize(self, text: str) -> list[str]:
        return str(text).split()

    # -------------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return [self._vocab.get(token, 0) for token in str(text).split()]

    # -------------------------------------------------------------------------
    def decode(self, token_ids: Any) -> str:
        ids = token_ids.ids if hasattr(token_ids, "ids") else token_ids
        return " ".join(
            self._id_to_token.get(int(token_id), "[UNK]") for token_id in ids
        )

    # -------------------------------------------------------------------------
    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self._id_to_token.get(int(token_id), "[UNK]") for token_id in token_ids]

    # -------------------------------------------------------------------------
    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

###############################################################################
def _benchmark_rows() -> list[tuple[int, str]]:
    return [
        (10, "alpha beta beta"),
        (11, "alpha gamma"),
        (12, "delta"),
    ]

###############################################################################
def run_deterministic_benchmark() -> dict[str, Any]:
    service = BenchmarkService()
    rows = _benchmark_rows()

    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: {
        "dummy/tokenizer": DummyTokenizer()
    }  # type: ignore[method-assign]
    service.calculate_morphological_consistency = (  # type: ignore[method-assign]
        lambda tokenizer, base_words: 0.5
    )

    result = service.run_benchmarks(
        dataset_name="custom/ds",
        tokenizer_ids=["dummy/tokenizer"],
        selected_metric_keys=None,
    )
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result

###############################################################################
def build_benchmark_metric_value_map(result: dict[str, Any]) -> dict[str, Any]:
    tokenizer_result = result["tokenizer_results"][0]
    efficiency_metrics = result["chart_data"]["efficiency"][0]
    metric_availability = result.get("runtime_metadata", {}).get(
        "metric_availability", {}
    )

    encode_tps = float(efficiency_metrics["value"])
    encode_cps = float(tokenizer_result["efficiency"]["encode_chars_per_second_mean"])
    wall_time_s = float(tokenizer_result["efficiency"]["end_to_end_wall_time_seconds"])
    exact_round_trip_rate = float(tokenizer_result["fidelity"]["exact_round_trip_rate"])
    normalized_round_trip_rate = float(
        tokenizer_result["fidelity"]["normalized_round_trip_rate"]
    )
    metric_values = {
        "eff.encode_tokens_per_second_mean": encode_tps,
        "eff.encode_tokens_per_second_ci95": 0.0,
        "eff.encode_chars_per_second_mean": encode_cps,
        "eff.end_to_end_wall_time_seconds": wall_time_s,
        "lat.encode_latency_p50_ms": float(
            tokenizer_result["latency"]["encode_latency_p50_ms"]
        ),
        "lat.encode_latency_p95_ms": float(
            tokenizer_result["latency"]["encode_latency_p95_ms"]
        ),
        "lat.encode_latency_p99_ms": float(
            tokenizer_result["latency"]["encode_latency_p99_ms"]
        ),
        "fid.exact_round_trip_rate": exact_round_trip_rate,
        "fid.normalized_round_trip_rate": normalized_round_trip_rate,
        "frag.pieces_per_word_mean": tokenizer_result["fragmentation"][
            "pieces_per_word_mean"
        ],
        "frag.tokens_per_character": tokenizer_result["fragmentation"][
            "tokens_per_character"
        ],
    }

    if metric_availability.get("unknown_token_rate"):
        metric_values["fid.unknown_token_rate"] = float(
            tokenizer_result["fidelity"]["unknown_token_rate"]
        )
    if metric_availability.get("vocab_character_overlap"):
        metric_values["fid.lossless_encodability_rate"] = tokenizer_result[
            "fidelity"
        ]["lossless_encodability_rate"]

    return metric_values

###############################################################################
def assert_metric_value(
    actual: Any, expected: Any, metric_key: str, path: str = ""
) -> None:
    location = f"{metric_key}{path}"
    if isinstance(expected, float):
        assert isinstance(actual, (int, float)), (
            f"{location} expected numeric float-like value, got {type(actual).__name__}: {actual}"
        )
        assert float(actual) == pytest.approx(expected, rel=1e-9, abs=1e-9), (
            f"{location} mismatch: expected {expected}, got {actual}"
        )
        return

    if isinstance(expected, int):
        assert actual == expected, (
            f"{location} mismatch: expected {expected}, got {actual}"
        )
        return

    if isinstance(expected, str):
        assert actual == expected, (
            f"{location} mismatch: expected {expected!r}, got {actual!r}"
        )
        return

    if isinstance(expected, list):
        assert isinstance(actual, list), (
            f"{location} expected list, got {type(actual).__name__}: {actual}"
        )
        assert len(actual) == len(expected), (
            f"{location} length mismatch: expected {len(expected)}, got {len(actual)}"
        )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            assert_metric_value(
                actual_item,
                expected_item,
                metric_key=metric_key,
                path=f"{path}[{index}]",
            )
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict), (
            f"{location} expected dict, got {type(actual).__name__}: {actual}"
        )
        assert set(actual.keys()) == set(expected.keys()), (
            f"{location} key mismatch: expected {sorted(expected.keys())}, got {sorted(actual.keys())}"
        )
        for key in sorted(expected.keys()):
            assert_metric_value(
                actual[key],
                expected[key],
                metric_key=metric_key,
                path=f"{path}.{key}",
            )
        return

    assert actual == expected, f"{location} mismatch: expected {expected}, got {actual}"
