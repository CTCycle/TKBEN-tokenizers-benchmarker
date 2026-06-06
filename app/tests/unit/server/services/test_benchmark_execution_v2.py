from __future__ import annotations

from typing import Any

import server.services.benchmarks as benchmarks_module
import server.services.benchmark_execution as benchmark_execution_module
from server.domain.benchmark_observations import BatchObservation
from server.domain.benchmarks import BenchmarkRunResponse
from server.services.benchmarks import BenchmarkService


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
        return " ".join(
            self._id_to_token.get(int(token_id), "[UNK]") for token_id in ids
        )

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self._id_to_token.get(int(token_id), "[UNK]") for token_id in token_ids]

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


class UnknownAwareTokenizer(DummyTokenizer):
    unk_token_id = 0

    def __init__(self) -> None:
        super().__init__()
        self._vocab = {"known": 1}
        self._id_to_token = {1: "known", 0: "[UNK]"}


def test_run_benchmarks_returns_v2_contract() -> None:
    service = BenchmarkService()
    rows = [
        (10, "alpha beta beta"),
        (11, "alpha gamma"),
        (12, "delta"),
    ]

    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: {
        "dummy/tokenizer": DummyTokenizer()
    }  # type: ignore[method-assign]
    service.calculate_morphological_consistency = (  # type: ignore[method-assign]
        lambda tokenizer, base_words: 0.5
    )

    original_perf_counter = benchmarks_module.time.perf_counter
    counter = {"calls": 0}

    def fake_perf_counter() -> float:
        counter["calls"] += 1
        return 100.0 + (counter["calls"] * 0.01)

    benchmarks_module.time.perf_counter = fake_perf_counter
    try:
        result = service.run_benchmarks(
            dataset_name="custom/ds",
            tokenizer_ids=["dummy/tokenizer"],
            selected_metric_keys=None,
        )
    finally:
        benchmarks_module.time.perf_counter = original_perf_counter

    assert isinstance(result, BenchmarkRunResponse)
    assert result.status == "success"
    assert result.dataset_name == "custom/ds"
    assert result.documents_processed == 3
    assert result.tokenizers_count == 1
    assert len(result.tokenizer_results) == 1
    assert result.tokenizer_results[0].status == "success"
    assert len(result.chart_data.efficiency) == 1
    assert len(result.per_document_stats) == 1
    assert result.tokenizer_results[0].tokenizer == "dummy/tokenizer"
    assert result.per_document_stats[0].tokenizer == "dummy/tokenizer"
    assert "benchmark_config" in result.runtime_metadata
    assert result.runtime_metadata["dataset_total_documents_available"] == 3
    assert result.runtime_metadata["dataset_documents_benchmarked"] == 3
    assert result.runtime_metadata["dataset_total_chars"] > 0
    assert result.runtime_metadata["dataset_total_utf8_bytes"] > 0
    assert len(result.runtime_metadata["tokenizer_metadata"]) == 1
    assert "benchmark_timing_boundaries" in result.runtime_metadata
    assert "metric_availability" in result.runtime_metadata
    assert result.runtime_metadata["metric_availability"]["resource_metrics"] is True
    assert (
        result.runtime_metadata["metric_availability"]["latency_distribution"] is True
    )
    assert result.runtime_metadata["metric_availability"]["per_document_stats"] is True
    assert result.runtime_metadata["end_to_end_benchmark_seconds"] >= 0.0
    assert result.tokenizer_results[0].efficiency.encode_only_wall_time_seconds >= 0.0
    assert (
        result.tokenizer_results[0].efficiency.dataset_stream_wall_time_seconds >= 0.0
    )
    assert result.tokenizer_results[0].efficiency.postprocess_wall_time_seconds >= 0.0


def test_run_benchmarks_enforces_max_documents_limit() -> None:
    service = BenchmarkService(max_documents=2)
    rows = [
        (1, "alpha beta"),
        (2, "gamma delta"),
        (3, "alpha"),
    ]

    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: {
        "dummy/tokenizer": DummyTokenizer()
    }  # type: ignore[method-assign]

    result = service.run_benchmarks(
        dataset_name="custom/ds",
        tokenizer_ids=["dummy/tokenizer"],
    )

    assert isinstance(result, BenchmarkRunResponse)
    assert result.documents_processed == 2
    assert len(result.per_document_stats[0].tokens_count) == 2


def test_run_benchmarks_isolates_tokenizer_failure() -> None:
    service = BenchmarkService()
    rows = [
        (10, "alpha beta"),
        (11, "gamma"),
    ]

    class BrokenTokenizer(DummyTokenizer):
        def encode(self, text: str) -> list[int]:
            raise RuntimeError("broken tokenizer")

    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: {  # type: ignore[method-assign]
        "ok/tokenizer": DummyTokenizer(),
        "broken/tokenizer": BrokenTokenizer(),
    }

    result = service.run_benchmarks(
        dataset_name="custom/ds",
        tokenizer_ids=["ok/tokenizer", "broken/tokenizer"],
    )

    assert isinstance(result, BenchmarkRunResponse)
    assert len(result.tokenizer_results) == 2
    assert {r.tokenizer for r in result.tokenizer_results} == {
        "ok/tokenizer",
        "broken/tokenizer",
    }
    assert "broken/tokenizer" in result.raw_observations
    assert result.raw_observations["broken/tokenizer"][0]["error"] == "RuntimeError"
    chart_tokenizers = {point.tokenizer for point in result.chart_data.efficiency}
    assert "ok/tokenizer" in chart_tokenizers
    assert "broken/tokenizer" not in chart_tokenizers


def test_run_benchmarks_uses_trial_level_speeds_for_ci() -> None:
    service = BenchmarkService()
    rows = [
        (1, "alpha beta"),
    ]

    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: {
        "dummy/tokenizer": DummyTokenizer()
    }  # type: ignore[method-assign]

    original_run_trials = benchmark_execution_module.run_tokenizer_trials

    def fake_run_trials(**kwargs: Any) -> list[BatchObservation]:
        return [
            BatchObservation("dummy/tokenizer", 0, 0, 1, 5, 100, 0, 100_000_000, 10.0),
            BatchObservation("dummy/tokenizer", 1, 0, 1, 5, 100, 0, 200_000_000, 10.0),
            BatchObservation("dummy/tokenizer", 2, 0, 1, 5, 100, 0, 400_000_000, 10.0),
        ]

    benchmark_execution_module.run_tokenizer_trials = fake_run_trials  # type: ignore[assignment]
    try:
        result = service.run_benchmarks(
            dataset_name="custom/ds",
            tokenizer_ids=["dummy/tokenizer"],
        )
    finally:
        benchmark_execution_module.run_tokenizer_trials = original_run_trials  # type: ignore[assignment]

    metrics = result.tokenizer_results[0].efficiency
    assert (
        metrics.encode_tokens_per_second_ci95_high
        > metrics.encode_tokens_per_second_ci95_low
    )


def test_run_benchmarks_uses_true_latency_distribution_five_number_summary() -> None:
    service = BenchmarkService()
    rows = [(1, "alpha beta")]

    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: {
        "dummy/tokenizer": DummyTokenizer()
    }  # type: ignore[method-assign]

    original_run_trials = benchmark_execution_module.run_tokenizer_trials

    def fake_run_trials(**kwargs: Any) -> list[BatchObservation]:
        return [
            BatchObservation(
                "dummy/tokenizer", 0, 0, 1, 5, 10, 0, 1_000_000, 10.0
            ),  # 1.0 ms
            BatchObservation(
                "dummy/tokenizer", 0, 1, 1, 5, 10, 0, 3_000_000, 10.0
            ),  # 3.0 ms
            BatchObservation(
                "dummy/tokenizer", 0, 2, 1, 5, 10, 0, 5_000_000, 10.0
            ),  # 5.0 ms
        ]

    benchmark_execution_module.run_tokenizer_trials = fake_run_trials  # type: ignore[assignment]
    try:
        result = service.run_benchmarks(
            dataset_name="custom/ds",
            tokenizer_ids=["dummy/tokenizer"],
        )
    finally:
        benchmark_execution_module.run_tokenizer_trials = original_run_trials  # type: ignore[assignment]

    dist = result.chart_data.latency_or_memory_distribution[0]
    assert dist.min == 1.0
    assert dist.max == 5.0
    assert dist.median == 3.0


def test_run_benchmarks_reports_utf8_bytes_throughput_and_unknown_rate() -> None:
    service = BenchmarkService()
    rows = [
        (1, "known é"),
        (2, "known 🙂 unknown"),
    ]

    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: {
        "ua/tokenizer": UnknownAwareTokenizer()
    }  # type: ignore[method-assign]

    result = service.run_benchmarks(
        dataset_name="custom/ds",
        tokenizer_ids=["ua/tokenizer"],
    )

    tokenizer_result = result.tokenizer_results[0]
    assert tokenizer_result.efficiency.encode_bytes_per_second_mean > 0.0
    assert tokenizer_result.fidelity.unknown_token_rate > 0.0
    assert (
        tokenizer_result.fragmentation.bytes_per_token
        >= tokenizer_result.fragmentation.characters_per_token
    )
    assert tokenizer_result.resources.peak_rss_mb > 0.0


def test_run_benchmarks_uses_utf8_bytes_per_token_for_per_doc_stats() -> None:
    service = BenchmarkService()
    rows = [(1, "known é")]
    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: {
        "ua/tokenizer": UnknownAwareTokenizer()
    }  # type: ignore[method-assign]

    result = service.run_benchmarks(
        dataset_name="custom/ds",
        tokenizer_ids=["ua/tokenizer"],
    )
    # "known é" => 8 UTF-8 bytes, 2 tokens
    assert result.per_document_stats[0].bytes_per_token[0] == 4.0


def test_run_benchmarks_can_disable_per_document_stats_and_persist_config() -> None:
    service = BenchmarkService()
    rows = [(1, "alpha beta"), (2, "gamma delta")]

    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: {
        "dummy/tokenizer": DummyTokenizer()
    }  # type: ignore[method-assign]

    result = service.run_benchmarks(
        dataset_name="custom/ds",
        tokenizer_ids=["dummy/tokenizer"],
        benchmark_config={
            "store_per_document_stats": False,
            "add_special_tokens": True,
            "padding": True,
            "truncation": True,
            "max_length": 32,
        },
    )

    assert result.per_document_stats == []
    assert result.config.store_per_document_stats is False
    assert result.config.add_special_tokens is True
    assert result.config.padding is True
    assert result.config.truncation is True
    assert result.config.max_length == 32
    assert result.runtime_metadata["metric_availability"]["per_document_stats"] is False


def test_run_benchmarks_returns_cancelled_status_when_stopped() -> None:
    service = BenchmarkService()
    rows = [(1, "alpha"), (2, "beta")]

    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: {
        "dummy/tokenizer": DummyTokenizer()
    }  # type: ignore[method-assign]

    calls = {"count": 0}

    def should_stop() -> bool:
        calls["count"] += 1
        return calls["count"] > 1

    result = service.run_benchmarks(
        dataset_name="custom/ds",
        tokenizer_ids=["dummy/tokenizer"],
        should_stop=should_stop,
    )

    assert result.status == "cancelled"


def test_run_benchmarks_all_failed_tokenizers_report_unavailable_metrics() -> None:
    service = BenchmarkService()
    rows = [(1, "alpha"), (2, "beta")]

    class BrokenTokenizer(DummyTokenizer):
        def encode(self, text: str) -> list[int]:
            raise RuntimeError("always broken")

    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: {
        "broken/tokenizer": BrokenTokenizer()
    }  # type: ignore[method-assign]

    result = service.run_benchmarks(
        dataset_name="custom/ds",
        tokenizer_ids=["broken/tokenizer"],
    )

    assert result.status == "success"
    assert len(result.tokenizer_results) == 1
    assert result.tokenizer_results[0].status == "failed"
    assert result.chart_data.efficiency == []
    assert result.chart_data.fidelity == []
    assert result.runtime_metadata["metric_availability"]["resource_metrics"] is False
    assert (
        result.runtime_metadata["metric_availability"]["latency_distribution"] is False
    )
    assert result.runtime_metadata["metric_availability"]["per_document_stats"] is False
