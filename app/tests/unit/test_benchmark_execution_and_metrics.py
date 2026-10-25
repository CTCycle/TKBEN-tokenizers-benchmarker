# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import threading
from typing import Any
import server.services.benchmark_execution as benchmark_execution_module
from server.services.benchmark_observations import BatchObservation
from server.contracts.benchmarks import BenchmarkRunResponse
from server.services.benchmarks import BenchmarkService
from server.services.benchmark_execution import normalize_vocabulary_token
from server.services.benchmark_observations import TokenizerRunConfig
from server.services.benchmark_engine import run_tokenizer_trials
from server.services.tokenizer_adapters import EncodedBatch
from server.services.benchmark_engine import summarize_observations
from server.services.benchmark_metric_plan import build_metric_plan
from server.services.benchmark_streams import iter_limited_rows
from server.services.benchmark_streams import iter_text_batches
from server.services.benchmark_streams import normalize_benchmark_text
from server.services.benchmark_spool import BenchmarkTextSpool
from server.services.benchmark_jobs import BenchmarkJobService
from pathlib import Path
import pytest
from server.contracts.benchmarks import BenchmarkRunRequest
from server.services.benchmarks import BENCHMARK_METRIC_CATALOG


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
    def _encode_one(self, text: str) -> list[int]:
        return [self._vocab.get(token, 0) for token in str(text).split()]

    # -------------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return self._encode_one(text)

    # -------------------------------------------------------------------------
    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, list[list[int]]]:
        del kwargs
        return {"input_ids": [self._encode_one(text) for text in texts]}

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
class UnknownAwareTokenizer(DummyTokenizer):
    unk_token_id = 0

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        super().__init__()
        self._vocab = {"known": 1}
        self._id_to_token = {1: "known", 0: "[UNK]"}


###############################################################################
def test_normalize_vocabulary_token_removes_only_top_level_markers() -> None:
    assert normalize_vocabulary_token("##tail") == "tail"
    assert normalize_vocabulary_token("▁hello") == "hello"
    assert normalize_vocabulary_token("Ġworld") == "world"
    assert normalize_vocabulary_token("Garden") == "Garden"
    assert normalize_vocabulary_token("word▁pieceĠtail##") == "word▁pieceĠtail##"


###############################################################################
def test_run_benchmarks_returns_contract() -> None:
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
    original_perf_counter = benchmark_execution_module.time.perf_counter
    counter = {"calls": 0}

    def fake_perf_counter() -> float:
        counter["calls"] += 1
        return 100.0 + (counter["calls"] * 0.01)

    benchmark_execution_module.time.perf_counter = fake_perf_counter
    try:
        result = service.run_benchmarks(
            dataset_name="custom/ds",
            tokenizer_ids=["dummy/tokenizer"],
            selected_metric_keys=None,
        )
    finally:
        benchmark_execution_module.time.perf_counter = original_perf_counter

    assert isinstance(result, BenchmarkRunResponse)
    assert result.status == "success"
    assert result.dataset_name == "custom/ds"
    assert result.documents_processed == 3
    assert result.tokenizers_count == 1
    assert len(result.tokenizer_results) == 1
    assert result.tokenizer_results[0].status == "success"
    assert len(result.dashboard.widgets) > 0
    assert len(result.per_document_stats) == 1
    assert result.tokenizer_results[0].tokenizer == "dummy/tokenizer"
    assert result.per_document_stats[0].tokenizer == "dummy/tokenizer"
    assert result.methodology_version == "semantic_honesty"
    assert "benchmark_config" in result.runtime_metadata
    assert result.runtime_metadata["dataset_total_documents_available"] == 3
    assert result.runtime_metadata["dataset_documents_benchmarked"] == 3
    assert result.runtime_metadata["dataset_total_chars"] > 0
    assert result.runtime_metadata["dataset_total_utf8_bytes"] > 0
    assert len(result.runtime_metadata["tokenizer_metadata"]) == 1
    assert "benchmark_timing_boundaries" in result.runtime_metadata
    assert result.runtime_metadata["end_to_end_benchmark_seconds"] >= 0.0
    assert result.tokenizer_results[0].efficiency.encode_only_wall_time_seconds >= 0.0
    assert (
        result.tokenizer_results[0].efficiency.dataset_stream_wall_time_seconds >= 0.0
    )
    assert result.tokenizer_results[0].efficiency.postprocess_wall_time_seconds >= 0.0
    assert result.tokenizer_results[0].fidelity.unknown_token_rate is None


###############################################################################
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


###############################################################################
def test_run_benchmarks_isolates_tokenizer_failure(monkeypatch) -> None:
    service = BenchmarkService()
    rows = [
        (10, "alpha beta"),
        (11, "gamma"),
    ]


    ###############################################################################
    class BrokenTokenizer(DummyTokenizer):

        # -------------------------------------------------------------------------
        def _encode_one(self, text: str) -> list[int]:
            raise RuntimeError("broken tokenizer")

    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: {  # type: ignore[method-assign]
        "ok/tokenizer": DummyTokenizer(),
        "broken/tokenizer": BrokenTokenizer(),
    }

    original_run_trials = benchmark_execution_module.run_tokenizer_trials
    start_barrier = threading.Barrier(2)

    def synchronized_run_trials(**kwargs: Any) -> list[BatchObservation]:
        start_barrier.wait(timeout=10)
        return original_run_trials(**kwargs)

    monkeypatch.setattr(
        benchmark_execution_module, "run_tokenizer_trials", synchronized_run_trials
    )

    result = service.run_benchmarks(
        dataset_name="custom/ds",
        tokenizer_ids=["ok/tokenizer", "broken/tokenizer"],
        benchmark_config={"parallelism": 2},
    )

    assert isinstance(result, BenchmarkRunResponse)
    assert len(result.tokenizer_results) == 2
    assert [r.tokenizer for r in result.tokenizer_results] == [
        "ok/tokenizer",
        "broken/tokenizer",
    ]
    assert [r.status for r in result.tokenizer_results] == ["success", "failed"]
    assert "broken/tokenizer" in result.raw_observations
    assert result.raw_observations["broken/tokenizer"][0]["error"] == "RuntimeError"
    efficiency_widget = next(
        widget
        for widget in result.dashboard.widgets
        if "eff.encode_tokens_per_second_mean" in widget.metric_keys
    )
    chart_tokenizers = {point.tokenizer for point in efficiency_widget.points}
    assert "ok/tokenizer" in chart_tokenizers
    assert "broken/tokenizer" not in chart_tokenizers


###############################################################################
def _service_with_tokenizers(tokenizers: dict[str, Any]) -> BenchmarkService:
    service = BenchmarkService()
    rows = [(10, "alpha beta"), (11, "gamma")]
    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: tokenizers  # type: ignore[method-assign]
    return service


###############################################################################
def test_run_benchmarks_parallelism_one_processes_tokenizers_serially() -> None:
    service = _service_with_tokenizers(
        {"first/tokenizer": DummyTokenizer(), "second/tokenizer": DummyTokenizer()}
    )

    result = service.run_benchmarks(
        dataset_name="custom/ds",
        tokenizer_ids=["first/tokenizer", "second/tokenizer"],
        benchmark_config={"parallelism": 1},
    )

    assert result.status == "success"
    assert [item.tokenizer for item in result.tokenizer_results] == [
        "first/tokenizer",
        "second/tokenizer",
    ]
    execution = result.runtime_metadata["benchmark_execution"]
    assert execution["requested_parallelism"] == 1
    assert execution["effective_parallelism"] == 1
    assert execution["tokenizer_count"] == 2
    assert execution["max_concurrent_workers_observed"] == 1


###############################################################################
def test_run_benchmarks_parallelism_two_overlaps_tokenizer_workloads(
    monkeypatch,
) -> None:
    service = _service_with_tokenizers(
        {"first/tokenizer": DummyTokenizer(), "second/tokenizer": DummyTokenizer()}
    )
    original_run_trials = benchmark_execution_module.run_tokenizer_trials
    start_barrier = threading.Barrier(2)

    def synchronized_run_trials(**kwargs: Any) -> list[BatchObservation]:
        start_barrier.wait(timeout=10)
        return original_run_trials(**kwargs)

    monkeypatch.setattr(
        benchmark_execution_module, "run_tokenizer_trials", synchronized_run_trials
    )
    result = service.run_benchmarks(
        dataset_name="custom/ds",
        tokenizer_ids=["first/tokenizer", "second/tokenizer"],
        benchmark_config={"parallelism": 2},
    )

    assert result.status == "success"
    assert (
        result.runtime_metadata["benchmark_execution"][
            "max_concurrent_workers_observed"
        ]
        == 2
    )


###############################################################################
def test_run_benchmarks_caps_workers_and_preserves_input_order(monkeypatch) -> None:


    ###############################################################################
    class DoubleTokenCountTokenizer(DummyTokenizer):

        # -------------------------------------------------------------------------
        def _encode_one(self, text: str) -> list[int]:
            encoded = super()._encode_one(text)
            return [token_id for token_id in encoded for _ in range(2)]

    service = _service_with_tokenizers(
        {
            "first/tokenizer": DummyTokenizer(),
            "second/tokenizer": DoubleTokenCountTokenizer(),
        }
    )
    original_execute = service._execute_tokenizer_workload
    second_completed = threading.Event()
    completion_order: list[str] = []
    completion_lock = threading.Lock()

    def finish_second_first(**kwargs: Any):
        tokenizer_name = kwargs["tokenizer_name"]
        if tokenizer_name == "first/tokenizer":
            assert second_completed.wait(timeout=10)
        result = original_execute(**kwargs)
        with completion_lock:
            completion_order.append(tokenizer_name)
        if tokenizer_name == "second/tokenizer":
            second_completed.set()
        return result

    monkeypatch.setattr(service, "_execute_tokenizer_workload", finish_second_first)
    progress: list[float] = []
    coordinator_thread = threading.get_ident()
    callback_threads: list[int] = []

    def record_progress(value: float) -> None:
        progress.append(value)
        callback_threads.append(threading.get_ident())

    result = service.run_benchmarks(
        dataset_name="custom/ds",
        tokenizer_ids=["first/tokenizer", "second/tokenizer"],
        benchmark_config={"parallelism": 128, "store_per_document_stats": True},
        progress_callback=record_progress,
    )

    assert result.status == "success"
    assert completion_order == ["second/tokenizer", "first/tokenizer"]
    assert [item.tokenizer for item in result.tokenizer_results] == [
        "first/tokenizer",
        "second/tokenizer",
    ]
    assert [item.tokenizer for item in result.per_document_stats] == [
        "first/tokenizer",
        "second/tokenizer",
    ]
    assert list(result.raw_observations) == ["first/tokenizer", "second/tokenizer"]
    first_tokens = sum(
        int(item["token_count"]) for item in result.raw_observations["first/tokenizer"]
    )
    second_tokens = sum(
        int(item["token_count"]) for item in result.raw_observations["second/tokenizer"]
    )
    assert second_tokens == first_tokens * 2
    assert result.per_document_stats[0].tokens_count == [2, 1]
    assert result.per_document_stats[1].tokens_count == [4, 2]
    execution = result.runtime_metadata["benchmark_execution"]
    assert result.config.parallelism == 128
    assert result.runtime_metadata["benchmark_config"]["parallelism"] == 128
    assert execution["requested_parallelism"] == 128
    assert execution["effective_parallelism"] == 2
    assert execution["tokenizer_count"] == 2
    assert execution["max_concurrent_workers_observed"] == 2
    assert progress == sorted(progress)
    assert progress[-1] == 99.0
    assert set(callback_threads) == {coordinator_thread}

    throughput_widget = next(
        widget
        for widget in result.dashboard.widgets
        if "eff.encode_tokens_per_second_mean" in widget.metric_keys
    )
    assert [point.tokenizer for point in throughput_widget.points] == [
        "first/tokenizer",
        "second/tokenizer",
    ]


###############################################################################
def test_run_benchmarks_cancellation_stops_scheduling_parallel_work(
    monkeypatch,
) -> None:
    service = _service_with_tokenizers(
        {
            "first/tokenizer": DummyTokenizer(),
            "second/tokenizer": DummyTokenizer(),
            "third/tokenizer": DummyTokenizer(),
        }
    )
    start_barrier = threading.Barrier(2)
    stop_requested = threading.Event()
    run_calls: list[str] = []
    calls_lock = threading.Lock()

    def cancel_after_both_start(**kwargs: Any) -> list[BatchObservation]:
        start_barrier.wait(timeout=10)
        with calls_lock:
            run_calls.append(kwargs["tokenizer"].tokenizer_id)
        stop_requested.set()
        return []

    monkeypatch.setattr(
        benchmark_execution_module, "run_tokenizer_trials", cancel_after_both_start
    )
    result = service.run_benchmarks(
        dataset_name="custom/ds",
        tokenizer_ids=[
            "first/tokenizer",
            "second/tokenizer",
            "third/tokenizer",
        ],
        benchmark_config={"parallelism": 2},
        should_stop=stop_requested.is_set,
    )

    assert result.status == "cancelled"
    assert len(run_calls) == 2
    assert set(run_calls) == {"first/tokenizer", "second/tokenizer"}
    assert result.runtime_metadata["benchmark_execution"]["effective_parallelism"] == 2


###############################################################################
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


###############################################################################
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

    latency_widget = next(
        widget
        for widget in result.dashboard.widgets
        if widget.widget_id == "benchmark.lat.encode_latency_distribution"
    )
    dist = latency_widget.distributions[0]
    assert dist.min == 1.0
    assert dist.max == 5.0
    assert dist.median == 3.0
    assert dist.sample_count == 3
    assert result.tokenizer_results[0].latency.sample_count == 3


###############################################################################
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


###############################################################################
def test_run_benchmarks_uses_all_timed_trials_for_latency_summary() -> None:
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
            BatchObservation("dummy/tokenizer", 0, 0, 1, 5, 10, 0, 1_000_000, 10.0),
            BatchObservation("dummy/tokenizer", 1, 0, 1, 5, 10, 0, 9_000_000, 10.0),
        ]

    benchmark_execution_module.run_tokenizer_trials = fake_run_trials  # type: ignore[assignment]
    try:
        result = service.run_benchmarks(
            dataset_name="custom/ds",
            tokenizer_ids=["dummy/tokenizer"],
        )
    finally:
        benchmark_execution_module.run_tokenizer_trials = original_run_trials  # type: ignore[assignment]

    latency = result.tokenizer_results[0].latency
    assert latency.encode_latency_p95_ms > 1.0
    assert latency.sample_count == 2


###############################################################################
def test_run_benchmarks_computes_real_fragmentation_buckets() -> None:
    service = BenchmarkService()
    rows = [(1, "a alpha alphabetic")]


    ###############################################################################
    class LengthSensitiveTokenizer(DummyTokenizer):

        # -------------------------------------------------------------------------
        def _encode_one(self, text: str) -> list[int]:
            token = str(text)
            if " " in token:
                return [
                    value for part in token.split() for value in self._encode_one(part)
                ]
            if len(token) <= 4:
                return [1]
            if len(token) <= 8:
                return [1, 2]
            return [1, 2, 3]

    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: {
        "dummy/tokenizer": LengthSensitiveTokenizer()
    }  # type: ignore[method-assign]

    result = service.run_benchmarks(
        dataset_name="custom/ds",
        tokenizer_ids=["dummy/tokenizer"],
    )

    buckets = {
        bucket.bucket: bucket.pieces_per_word_mean
        for bucket in result.tokenizer_results[
            0
        ].fragmentation.fragmentation_by_word_length_bucket
    }
    assert set(buckets) == {"short_1_4", "medium_5_8", "long_9_plus"}
    assert len(set(round(value, 6) for value in buckets.values())) > 1


###############################################################################
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


###############################################################################
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
        },
    )

    assert len(result.per_document_stats) == 1
    assert result.config.store_per_document_stats is False


###############################################################################
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


###############################################################################
def test_run_benchmarks_all_failed_tokenizers_report_unavailable_metrics() -> None:
    service = BenchmarkService()
    rows = [(1, "alpha"), (2, "beta")]


    ###############################################################################
    class BrokenTokenizer(DummyTokenizer):

        # -------------------------------------------------------------------------
        def _encode_one(self, text: str) -> list[int]:
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
    assert result.dashboard.widgets == []


###############################################################################
class CountingAdapter:
    tokenizer_id = "dummy"

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.calls = 0

    # -------------------------------------------------------------------------
    def encode_batch(self, texts, **kwargs) -> EncodedBatch:  # type: ignore[no-untyped-def]
        del kwargs
        self.calls += 1
        return EncodedBatch(
            token_counts=[len(text) for text in texts],
            unknown_counts=[0 for _ in texts],
            input_ids_by_doc=[[1 for _ in text] for text in texts],
        )


###############################################################################
def test_warmup_excluded_and_timed_trials_control_observations() -> None:
    adapter = CountingAdapter()
    batches = [["a", "bb"], ["ccc"]]
    observations = run_tokenizer_trials(
        tokenizer=adapter,
        text_batches_factory=lambda: batches,
        config=TokenizerRunConfig(batch_size=2),
        warmup_trials=2,
        timed_trials=3,
    )
    assert len(observations) == 6
    assert adapter.calls == (2 * 2) + (3 * 2)


###############################################################################
def test_run_tokenizer_trials_respects_cancellation() -> None:
    adapter = CountingAdapter()
    batches = [["a"], ["b"]]
    stop = {"calls": 0}

    def should_stop() -> bool:
        stop["calls"] += 1
        return stop["calls"] > 1

    observations = run_tokenizer_trials(
        tokenizer=adapter,
        text_batches_factory=lambda: batches,
        config=TokenizerRunConfig(batch_size=1),
        warmup_trials=0,
        timed_trials=5,
        should_stop=should_stop,
    )
    assert len(observations) <= 1


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


###############################################################################
class MemoryAdapter:
    tokenizer_id = "dummy"

    # -------------------------------------------------------------------------
    def encode_batch(self, texts, **kwargs) -> EncodedBatch:  # type: ignore[no-untyped-def]
        del kwargs
        return EncodedBatch(
            token_counts=[len(text) for text in texts],
            unknown_counts=[0 for _ in texts],
            input_ids_by_doc=[[ord(character) for character in text] for text in texts],
        )


###############################################################################
def test_memory_peak_is_measured_not_placeholder() -> None:
    observations = run_tokenizer_trials(
        tokenizer=MemoryAdapter(),
        text_batches_factory=lambda: [["a", "b"]],
        config=TokenizerRunConfig(batch_size=2),
        warmup_trials=0,
        timed_trials=1,
    )
    assert observations
    assert observations[0].peak_rss_mb is not None
    assert float(observations[0].peak_rss_mb or 0.0) > 0.0


###############################################################################
def test_metric_plan_flags_selected_metrics() -> None:
    plan = build_metric_plan(
        [
            "lat.encode_latency_p95_ms",
            "fid.exact_round_trip_rate",
            "fid.unknown_token_rate",
        ],
        store_per_document_stats=False,
    )
    assert plan.needs_latency is True
    assert plan.needs_fidelity is True
    assert plan.needs_round_trip is True
    assert plan.needs_unknown_rate is True
    assert plan.needs_per_document_stats is False


###############################################################################
def test_normalize_benchmark_text_handles_none_and_coercion() -> None:
    assert normalize_benchmark_text(None) is None
    assert normalize_benchmark_text("abc") == "abc"
    assert normalize_benchmark_text(123) == "123"


###############################################################################
def test_iter_limited_rows_respects_limit_and_skips_none() -> None:
    rows = [(1, "a"), (2, None), (3, "b"), (4, "c")]
    limited = list(iter_limited_rows(rows, 3))
    assert limited == [(1, "a"), (3, "b")]


###############################################################################
def test_iter_text_batches_groups_rows_by_batch_size() -> None:
    rows = [(1, "a"), (2, "b"), (3, "c")]
    batches = list(iter_text_batches(rows, 2))
    assert batches == [[(1, "a"), (2, "b")], [(3, "c")]]


###############################################################################
def test_benchmark_text_spool_replays_rows_and_batches() -> None:
    spool = BenchmarkTextSpool()
    spool.append(1, "alpha")
    spool.append(2, "beta")
    spool.append(3, "gamma")
    spool.finalize()

    assert list(spool.iter_rows()) == [(1, "alpha"), (2, "beta"), (3, "gamma")]
    assert list(spool.iter_text_batches(2)) == [["alpha", "beta"], ["gamma"]]
    spool.cleanup()


###############################################################################
def test_benchmark_text_spool_cleanup_removes_file() -> None:
    spool = BenchmarkTextSpool()
    path = spool.path
    spool.append(1, "x")
    spool.finalize()
    assert path.exists()
    spool.cleanup()
    assert not path.exists()


###############################################################################
class DummyJobManager:

    # -------------------------------------------------------------------------
    def __init__(self, *, stopped: bool = False) -> None:
        self.stopped = stopped

    # -------------------------------------------------------------------------
    def should_stop(self, job_id: str) -> bool:
        del job_id
        return self.stopped

    # -------------------------------------------------------------------------
    def update_progress(self, job_id: str, value: float) -> None:
        del job_id, value


###############################################################################
def test_run_benchmark_job_builds_and_saves_report(monkeypatch) -> None:


    ###############################################################################
    class FakeBenchmarkService:

        # -------------------------------------------------------------------------
        def __init__(self, max_documents: int = 0):
            self.max_documents = max_documents

        # -------------------------------------------------------------------------
        def run_benchmarks(self, **kwargs):
            del kwargs
            return {
                "status": "success",
                "schema_version": 3,
                "methodology_version": "semantic_honesty",
                "dataset_name": "custom/sample",
                "documents_processed": 2,
                "tokenizers_processed": ["bert-base-uncased"],
                "tokenizers_count": 1,
                "config": {
                    "max_documents": 0,
                    "warmup_trials": 2,
                    "timed_trials": 8,
                    "batch_size": 16,
                    "seed": 42,
                    "parallelism": 1,
                    "add_special_tokens": False,
                    "padding": False,
                    "truncation": False,
                    "max_length": None,
                    "store_per_document_stats": True,
                    "per_document_sample_size": 500,
                },
                "hardware_profile": {
                    "runtime": "3.14",
                    "os": "test",
                    "cpu_model": None,
                    "cpu_logical_cores": None,
                    "memory_total_mb": None,
                },
                "trial_summary": {"warmup_trials": 2, "timed_trials": 8},
                "tokenizer_results": [],
                "dashboard": {"widgets": []},
                "per_document_stats": [],
                "runtime_metadata": {},
                "raw_observations": {},
            }

        # -------------------------------------------------------------------------
        def save_benchmark_report(self, payload):
            assert payload["dataset_name"] == "custom/sample"
            return 11


    ###############################################################################
    class FakeBenchmarkReportService:

        # -------------------------------------------------------------------------
        def save_benchmark_report(self, payload):
            assert payload["dataset_name"] == "custom/sample"
            return 11

    monkeypatch.setattr(
        "server.services.benchmark_jobs.BenchmarkService", FakeBenchmarkService
    )
    monkeypatch.setattr(
        "server.services.benchmark_jobs.BenchmarkReportService",
        FakeBenchmarkReportService,
    )

    service = BenchmarkJobService()
    result = service.run_benchmark_job(
        request_payload={
            "dataset_name": "custom/sample",
            "tokenizers": ["bert-base-uncased"],
            "config": {},
        },
        job_manager=DummyJobManager(),
        job_id="job1",
    )

    assert result["report_id"] == 11
    assert result["report_version"] == 5


###############################################################################
def test_run_benchmark_job_returns_cancelled_payload_without_persist(
    monkeypatch,
) -> None:


    ###############################################################################
    class FakeBenchmarkService:

        # -------------------------------------------------------------------------
        def __init__(self, max_documents: int = 0):
            self.max_documents = max_documents

        # -------------------------------------------------------------------------
        def run_benchmarks(self, **kwargs):
            del kwargs
            return {
                "status": "cancelled",
                "schema_version": 3,
                "methodology_version": "semantic_honesty",
                "dataset_name": "custom/sample",
                "documents_processed": 0,
                "tokenizers_processed": [],
                "tokenizers_count": 0,
                "config": {
                    "max_documents": 0,
                    "warmup_trials": 2,
                    "timed_trials": 8,
                    "batch_size": 16,
                    "seed": 42,
                    "parallelism": 1,
                    "add_special_tokens": False,
                    "padding": False,
                    "truncation": False,
                    "max_length": None,
                    "store_per_document_stats": True,
                    "per_document_sample_size": 500,
                },
                "hardware_profile": {
                    "runtime": "",
                    "os": "",
                    "cpu_model": None,
                    "cpu_logical_cores": None,
                    "memory_total_mb": None,
                },
                "trial_summary": {"warmup_trials": 0, "timed_trials": 0},
                "tokenizer_results": [],
                "dashboard": {"widgets": []},
                "per_document_stats": [],
                "runtime_metadata": {},
                "raw_observations": {},
            }

        # -------------------------------------------------------------------------
        def save_benchmark_report(self, payload):
            raise AssertionError(
                "save_benchmark_report must not be called for cancelled runs"
            )

    monkeypatch.setattr(
        "server.services.benchmark_jobs.BenchmarkService", FakeBenchmarkService
    )

    service = BenchmarkJobService()
    result = service.run_benchmark_job(
        request_payload={
            "dataset_name": "custom/sample",
            "tokenizers": ["bert-base-uncased"],
            "config": {},
        },
        job_manager=DummyJobManager(stopped=True),
        job_id="job1",
    )

    assert result["status"] == "cancelled"
    assert result["report_id"] is None


###############################################################################
class FakeBenchmarkRepository:

    # -------------------------------------------------------------------------
    def get_dataset_document_count(self, dataset_name: str) -> int:
        return 7 if dataset_name == "custom/sample" else 0

    # -------------------------------------------------------------------------
    def get_missing_persisted_tokenizers(self, tokenizer_ids: list[str]) -> list[str]:
        return [name for name in tokenizer_ids if name != "bert-base-uncased"]

    # -------------------------------------------------------------------------
    def get_tokenizer_sources(self, tokenizer_ids: list[str]) -> dict[str, str]:
        return {
            name: "huggingface" for name in tokenizer_ids if name == "bert-base-uncased"
        }


###############################################################################
def test_benchmark_service_uses_repository_for_dataset_and_tokenizer_checks() -> None:
    service = BenchmarkService()
    service.repository = FakeBenchmarkRepository()  # type: ignore[assignment]

    assert service.get_dataset_document_count("custom/sample") == 7
    assert service.get_dataset_document_count("missing") == 0

    missing = service.get_missing_persisted_tokenizers(["bert-base-uncased", "missing"])
    assert "missing" in missing


###############################################################################
def test_benchmark_service_preserves_repository_missing_with_cached_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("server.services.benchmarks.TOKENIZERS_PATH", tmp_path)
    cached_dir = tmp_path / "missing"
    cached_dir.mkdir()
    (cached_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    service = BenchmarkService()
    service.repository = FakeBenchmarkRepository()  # type: ignore[assignment]

    missing = service.get_missing_persisted_tokenizers(["missing"])

    assert missing == ["missing"]


###############################################################################
def test_prepare_run_owns_admission_checks_and_normalizes_job_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = BenchmarkService()
    monkeypatch.setattr(service, "get_dataset_document_count", lambda dataset_name: 3)
    monkeypatch.setattr(
        service, "get_missing_persisted_tokenizers", lambda tokenizers: []
    )

    payload = BenchmarkRunRequest(
        tokenizers=["CUSTOM_demo"],
        dataset_name="custom/sample",
        config={"max_documents": 2},
    )
    prepared = service.prepare_run(payload)

    assert prepared["dataset_name"] == "custom/sample"
    assert prepared["config"]["max_documents"] == 2


###############################################################################


CORE_TOKENIZER_BENCHMARK_METRIC_KEYS = {
    metric["key"]
    for category in BENCHMARK_METRIC_CATALOG
    for metric in category.get("metrics", [])
    if isinstance(metric, dict)
    and isinstance(metric.get("key"), str)
    and bool(metric.get("core"))
}


###############################################################################
class DummyTokenizer__tokenizer_benchmark_metrics:
    name_or_path = "dummy/tokenizer"

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self._vocab = {
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
    def _encode_one(self, text: str) -> list[int]:
        return [self._vocab.get(token, 0) for token in str(text).split()]

    # -------------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return self._encode_one(text)

    # -------------------------------------------------------------------------
    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, list[list[int]]]:
        del kwargs
        return {"input_ids": [self._encode_one(text) for text in texts]}

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
def run_deterministic_benchmark() -> dict[str, Any]:
    service = BenchmarkService()
    rows = [
        (10, "alpha beta beta"),
        (11, "alpha gamma"),
        (12, "delta"),
    ]
    service.get_dataset_document_count = lambda dataset_name: len(rows)  # type: ignore[method-assign]
    service.stream_dataset_rows_from_database = lambda dataset_name: iter(rows)  # type: ignore[method-assign]
    service.load_tokenizers = lambda tokenizer_ids: {
        "dummy/tokenizer": DummyTokenizer__tokenizer_benchmark_metrics()
    }  # type: ignore[method-assign]
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
    widgets = result["dashboard"]["widgets"]
    efficiency_widget = next(
        widget
        for widget in widgets
        if "eff.encode_tokens_per_second_mean" in widget["metric_keys"]
    )
    efficiency_metrics = efficiency_widget["points"][0]
    metric_values = {
        "meta.vocabulary_size": int(tokenizer_result["vocabulary_size"]),
        "eff.encode_tokens_per_second_mean": float(efficiency_metrics["value"]),
        "eff.encode_tokens_per_second_ci95": 0.0,
        "eff.encode_chars_per_second_mean": float(
            tokenizer_result["efficiency"]["encode_chars_per_second_mean"]
        ),
        "eff.end_to_end_wall_time_seconds": float(
            tokenizer_result["efficiency"]["end_to_end_wall_time_seconds"]
        ),
        "lat.encode_latency_p50_ms": float(
            tokenizer_result["latency"]["encode_latency_p50_ms"]
        ),
        "lat.encode_latency_p95_ms": float(
            tokenizer_result["latency"]["encode_latency_p95_ms"]
        ),
        "lat.encode_latency_p99_ms": float(
            tokenizer_result["latency"]["encode_latency_p99_ms"]
        ),
        "fid.exact_round_trip_rate": float(
            tokenizer_result["fidelity"]["exact_round_trip_rate"]
        ),
        "fid.normalized_round_trip_rate": float(
            tokenizer_result["fidelity"]["normalized_round_trip_rate"]
        ),
        "frag.pieces_per_word_mean": tokenizer_result["fragmentation"][
            "pieces_per_word_mean"
        ],
        "frag.tokens_per_character": tokenizer_result["fragmentation"][
            "tokens_per_character"
        ],
    }
    if tokenizer_result["fidelity"]["unknown_token_rate"] is not None:
        metric_values["fid.unknown_token_rate"] = float(
            tokenizer_result["fidelity"]["unknown_token_rate"]
        )
    if tokenizer_result["fidelity"]["lossless_encodability_rate"] is not None:
        metric_values["fid.lossless_encodability_rate"] = float(
            tokenizer_result["fidelity"]["lossless_encodability_rate"]
        )
    return metric_values


###############################################################################
@pytest.fixture(scope="module")
def benchmark_metric_values() -> dict[str, object]:
    result = run_deterministic_benchmark()
    return build_benchmark_metric_value_map(result)


###############################################################################
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
