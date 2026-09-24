from __future__ import annotations

import threading
from typing import Any

import server.services.benchmark_execution as benchmark_execution_module
from server.services.benchmark_observations import BatchObservation
from server.contracts.benchmarks import BenchmarkRunResponse
from server.services.benchmarks import BenchmarkService
from server.services.benchmark_execution import normalize_vocabulary_token

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
        def encode(self, text: str) -> list[int]:
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
    assert result.runtime_metadata["benchmark_execution"][
        "max_concurrent_workers_observed"
    ] == 2

###############################################################################
def test_run_benchmarks_caps_workers_and_preserves_input_order(monkeypatch) -> None:
    class DoubleTokenCountTokenizer(DummyTokenizer):
        def encode(self, text: str) -> list[int]:
            encoded = super().encode(text)
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
        int(item["token_count"])
        for item in result.raw_observations["first/tokenizer"]
    )
    second_tokens = sum(
        int(item["token_count"])
        for item in result.raw_observations["second/tokenizer"]
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
    assert result.runtime_metadata["benchmark_execution"][
        "effective_parallelism"
    ] == 2

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
        def encode(self, text: str) -> list[int]:
            token = str(text)
            if " " in token:
                return [value for part in token.split() for value in self.encode(part)]
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
    assert result.dashboard.widgets == []
