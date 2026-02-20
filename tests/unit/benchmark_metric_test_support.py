from __future__ import annotations

from typing import Any

import pytest

import TKBEN.server.services.benchmarks as benchmarks_module
from TKBEN.server.services.benchmarks import BENCHMARK_METRIC_CATALOG, BenchmarkService


BENCHMARK_METRIC_KEYS: set[str] = {
    metric["key"]
    for category in BENCHMARK_METRIC_CATALOG
    for metric in category.get("metrics", [])
    if isinstance(metric, dict) and isinstance(metric.get("key"), str)
}

EXPECTED_BENCHMARK_METRIC_VALUES: dict[str, Any] = {
    "run.dataset_name": "custom/ds",
    "run.documents_processed": 3,
    "run.tokenizers_count": 1,
    "run.tokenizers_processed": ["dummy/tokenizer"],
    "global.tokenization_speed_tps": 1.5,
    "global.throughput_chars_per_sec": 7.75,
    "global.vocabulary_size": 5,
    "global.avg_sequence_length": 2.0,
    "global.median_sequence_length": 2.0,
    "global.subword_fertility": 1.0,
    "global.oov_rate": 0.0,
    "global.word_recovery_rate": 100.0,
    "global.character_coverage": 90.9090909090909,
    "global.determinism_rate": 1.0,
    "global.boundary_preservation_rate": 0.6666666666666666,
    "global.round_trip_fidelity_rate": 1.0,
    "vocabulary.vocabulary_size": 5,
    "vocabulary.subwords_count": 1,
    "vocabulary.true_words_count": 4,
    "vocabulary.subwords_percentage": 20.0,
    "vocabulary.token_length_distribution": [
        {"bin_start": 0, "bin_end": 2, "count": 0},
        {"bin_start": 2, "bin_end": 4, "count": 0},
        {"bin_start": 4, "bin_end": 6, "count": 5},
    ],
    "speed.tokens_per_second": 1.5,
    "speed.chars_per_second": 7.75,
    "speed.processing_time_seconds": 4.0,
    "internal.model_size_mb": 0.0,
    "internal.segmentation_consistency": 0.5,
    "internal.token_distribution_entropy": 1.9182958340544893,
    "internal.rare_token_tail_1": 2,
    "internal.rare_token_tail_2": 2,
    "internal.compression_chars_per_token": 5.166666666666667,
    "internal.compression_bytes_per_character": 1.0,
    "internal.round_trip_text_fidelity_rate": 1.0,
    "internal.token_id_ordering_monotonicity": 0.75,
    "internal.token_unigram_coverage": 0.8,
    "document.tokens_count": [3, 2, 1],
    "document.tokens_to_words_ratio": [1.0, 1.0, 1.0],
    "document.bytes_per_token": [5.0, 5.5, 5.0],
    "document.boundary_preservation_rate": [1.0, 1.0, 0.0],
    "document.round_trip_token_fidelity": [1.0, 1.0, 1.0],
    "document.round_trip_text_fidelity": [1.0, 1.0, 1.0],
    "document.determinism_stability": [1.0, 1.0, 1.0],
    "document.bytes_per_character": [1.0, 1.0, 1.0],
}

DATASET_BENCHMARK_METRIC_KEYS: set[str] = {
    "run.dataset_name",
    "run.documents_processed",
    "run.tokenizers_count",
    "run.tokenizers_processed",
    "speed.tokens_per_second",
    "speed.chars_per_second",
    "speed.processing_time_seconds",
    "document.tokens_count",
    "document.tokens_to_words_ratio",
    "document.bytes_per_token",
    "document.boundary_preservation_rate",
    "document.round_trip_token_fidelity",
    "document.round_trip_text_fidelity",
    "document.determinism_stability",
    "document.bytes_per_character",
    "global.tokenization_speed_tps",
    "global.throughput_chars_per_sec",
    "global.avg_sequence_length",
    "global.median_sequence_length",
    "global.subword_fertility",
    "global.oov_rate",
    "global.word_recovery_rate",
    "global.character_coverage",
    "global.determinism_rate",
    "global.boundary_preservation_rate",
    "global.round_trip_fidelity_rate",
}

TOKENIZER_BENCHMARK_METRIC_KEYS: set[str] = BENCHMARK_METRIC_KEYS - DATASET_BENCHMARK_METRIC_KEYS


class DummyTokenizer:
    name_or_path = "dummy/tokenizer"

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {
            "alpha": 1,
            "beta": 2,
            "gamma": 3,
            "delta": 4,
            "##tail": 5,
        }
        self._id_to_token = {value: key for key, value in self._vocab.items()}

    def tokenize(self, text: str) -> list[str]:
        return str(text).split()

    def encode(self, text: str) -> list[int]:
        return [self._vocab.get(token, 0) for token in str(text).split()]

    def decode(self, token_ids: Any) -> str:
        ids = token_ids.ids if hasattr(token_ids, "ids") else token_ids
        return " ".join(self._id_to_token.get(int(token_id), "[UNK]") for token_id in ids)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self._id_to_token.get(int(token_id), "[UNK]") for token_id in token_ids]

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


def _benchmark_rows() -> list[tuple[int, str]]:
    return [
        (10, "alpha beta beta"),
        (11, "alpha gamma"),
        (12, "delta"),
    ]


def run_deterministic_benchmark() -> dict[str, Any]:
    service = BenchmarkService()
    rows = _benchmark_rows()

    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: {"dummy/tokenizer": DummyTokenizer()}  # type: ignore[method-assign]
    service.persist_results = lambda **kwargs: None  # type: ignore[method-assign]
    service.calculate_morphological_consistency = (  # type: ignore[method-assign]
        lambda tokenizer, base_words: 0.5
    )

    original_perf_counter = benchmarks_module.time.perf_counter
    counter = {"calls": 0}

    def fake_perf_counter() -> float:
        counter["calls"] += 1
        return 100.0 if counter["calls"] == 1 else 104.0

    benchmarks_module.time.perf_counter = fake_perf_counter
    try:
        return service.run_benchmarks(
            dataset_name="custom/ds",
            tokenizer_ids=["dummy/tokenizer"],
            selected_metric_keys=None,
        )
    finally:
        benchmarks_module.time.perf_counter = original_perf_counter


def build_benchmark_metric_value_map(result: dict[str, Any]) -> dict[str, Any]:
    global_metrics = result["global_metrics"][0]
    speed_metrics = result["speed_metrics"][0]
    vocabulary_stats = result["vocabulary_stats"][0]
    per_document_stats = result["per_document_stats"][0]
    token_length_distribution = result["token_length_distributions"][0]

    return {
        "run.dataset_name": result["dataset_name"],
        "run.documents_processed": result["documents_processed"],
        "run.tokenizers_count": result["tokenizers_count"],
        "run.tokenizers_processed": result["tokenizers_processed"],
        "global.tokenization_speed_tps": global_metrics["tokenization_speed_tps"],
        "global.throughput_chars_per_sec": global_metrics["throughput_chars_per_sec"],
        "global.vocabulary_size": global_metrics["vocabulary_size"],
        "global.avg_sequence_length": global_metrics["avg_sequence_length"],
        "global.median_sequence_length": global_metrics["median_sequence_length"],
        "global.subword_fertility": global_metrics["subword_fertility"],
        "global.oov_rate": global_metrics["oov_rate"],
        "global.word_recovery_rate": global_metrics["word_recovery_rate"],
        "global.character_coverage": global_metrics["character_coverage"],
        "global.determinism_rate": global_metrics["determinism_rate"],
        "global.boundary_preservation_rate": global_metrics["boundary_preservation_rate"],
        "global.round_trip_fidelity_rate": global_metrics["round_trip_fidelity_rate"],
        "vocabulary.vocabulary_size": vocabulary_stats["vocabulary_size"],
        "vocabulary.subwords_count": vocabulary_stats["subwords_count"],
        "vocabulary.true_words_count": vocabulary_stats["true_words_count"],
        "vocabulary.subwords_percentage": vocabulary_stats["subwords_percentage"],
        "vocabulary.token_length_distribution": token_length_distribution["bins"],
        "speed.tokens_per_second": speed_metrics["tokens_per_second"],
        "speed.chars_per_second": speed_metrics["chars_per_second"],
        "speed.processing_time_seconds": speed_metrics["processing_time_seconds"],
        "internal.model_size_mb": global_metrics["model_size_mb"],
        "internal.segmentation_consistency": global_metrics["segmentation_consistency"],
        "internal.token_distribution_entropy": global_metrics["token_distribution_entropy"],
        "internal.rare_token_tail_1": global_metrics["rare_token_tail_1"],
        "internal.rare_token_tail_2": global_metrics["rare_token_tail_2"],
        "internal.compression_chars_per_token": global_metrics["compression_chars_per_token"],
        "internal.compression_bytes_per_character": global_metrics[
            "compression_bytes_per_character"
        ],
        "internal.round_trip_text_fidelity_rate": global_metrics[
            "round_trip_text_fidelity_rate"
        ],
        "internal.token_id_ordering_monotonicity": global_metrics[
            "token_id_ordering_monotonicity"
        ],
        "internal.token_unigram_coverage": global_metrics["token_unigram_coverage"],
        "document.tokens_count": per_document_stats["tokens_count"],
        "document.tokens_to_words_ratio": per_document_stats["tokens_to_words_ratio"],
        "document.bytes_per_token": per_document_stats["bytes_per_token"],
        "document.boundary_preservation_rate": per_document_stats[
            "boundary_preservation_rate"
        ],
        "document.round_trip_token_fidelity": per_document_stats[
            "round_trip_token_fidelity"
        ],
        "document.round_trip_text_fidelity": per_document_stats[
            "round_trip_text_fidelity"
        ],
        "document.determinism_stability": per_document_stats["determinism_stability"],
        "document.bytes_per_character": per_document_stats["bytes_per_character"],
    }


def assert_metric_value(actual: Any, expected: Any, metric_key: str, path: str = "") -> None:
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
        assert actual == expected, f"{location} mismatch: expected {expected}, got {actual}"
        return

    if isinstance(expected, str):
        assert actual == expected, f"{location} mismatch: expected {expected!r}, got {actual!r}"
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
