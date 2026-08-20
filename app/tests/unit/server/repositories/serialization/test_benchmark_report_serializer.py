from __future__ import annotations

from sqlalchemy import create_engine
from datetime import datetime, timezone
from sqlalchemy.orm import Session
import pytest

from server.contracts.benchmarks import BenchmarkReportQuery
from server.repositories.database.backend import get_database
from server.repositories.schemas.models import Base, Dataset
from server.repositories.serialization.benchmark_reports import (
    BenchmarkReportSerializer,
)

###############################################################################
def _build_payload(dataset_name: str) -> dict:
    return {
        "status": "success",
        "schema_version": 3,
        "methodology_version": "semantic_honesty",
        "report_version": 5,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_name": "serializer test",
        "selected_metric_keys": ["eff.encode_tokens_per_second_mean"],
        "dataset_name": dataset_name,
        "documents_processed": 2,
        "tokenizers_processed": ["dummy/tokenizer"],
        "tokenizers_count": 1,
        "config": {
            "max_documents": 0,
            "warmup_trials": 2,
            "timed_trials": 8,
            "batch_size": 16,
            "seed": 42,
            "parallelism": 1,
            "include_lm_metrics": False,
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
        "tokenizer_results": [
            {
                "tokenizer": "dummy/tokenizer",
                "status": "success",
                "error_type": None,
                "error_message": None,
                "tokenizer_family": "unknown",
                "runtime_backend": "transformers_auto",
                "vocabulary_size": 10,
                "added_tokens": 0,
                "special_token_share": 0.0,
                "efficiency": {
                    "encode_tokens_per_second_mean": 10.0,
                    "encode_tokens_per_second_ci95_low": 9.0,
                    "encode_tokens_per_second_ci95_high": 11.0,
                    "encode_chars_per_second_mean": 100.0,
                    "encode_bytes_per_second_mean": 100.0,
                    "encode_only_wall_time_seconds": 1.0,
                    "dataset_stream_wall_time_seconds": 0.1,
                    "postprocess_wall_time_seconds": 0.2,
                    "end_to_end_wall_time_seconds": 1.3,
                    "load_time_seconds": 0.0,
                },
                "latency": {
                    "encode_latency_p50_ms": 1.0,
                    "encode_latency_p95_ms": 2.0,
                    "encode_latency_p99_ms": 3.0,
                    "sample_count": 8,
                },
                "fidelity": {
                    "exact_round_trip_rate": 1.0,
                    "normalized_round_trip_rate": 1.0,
                    "unknown_token_rate": 0.0,
                    "byte_fallback_rate": None,
                    "lossless_encodability_rate": 100.0,
                },
                "fragmentation": {
                    "tokens_per_character": 0.5,
                    "characters_per_token": 2.0,
                    "tokens_per_byte": 0.5,
                    "bytes_per_token": 2.0,
                    "pieces_per_word_mean": 1.0,
                    "fragmentation_by_word_length_bucket": [],
                },
                "resources": {
                    "peak_rss_mb": 50.0,
                    "memory_delta_mb": 1.0,
                },
            }
        ],
        "dashboard": {"widgets": []},
        "per_document_stats": [],
        "runtime_metadata": {
            "python_version": "x",
        },
        "raw_observations": {},
    }

###############################################################################
def test_benchmark_report_serializer_round_trip(monkeypatch) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)

    dataset_name = "custom/serializer_ds"
    with Session(bind=engine) as session:
        now = datetime.now(timezone.utc)
        session.add(Dataset(name=dataset_name, status="ready", created_at=now, updated_at=now, ready_at=now))
        session.commit()

    serializer = BenchmarkReportSerializer()
    payload = _build_payload(dataset_name)

    report_id = serializer.save_benchmark_report(payload)
    stored = serializer.load_benchmark_report_by_id(report_id)
    summaries = serializer.list_benchmark_reports(BenchmarkReportQuery(limit=10))

    assert stored is not None
    assert stored["report_id"] == report_id
    assert stored["dataset_name"] == dataset_name
    assert stored["tokenizers_count"] == 1
    assert stored["tokenizer_results"][0]["status"] == "success"
    assert (
        stored["tokenizer_results"][0]["efficiency"]["encode_only_wall_time_seconds"]
        == 1.0
    )
    assert summaries.reports[0].report_id == report_id
    assert summaries.reports[0].dataset_name == dataset_name
    assert summaries.total == 1

###############################################################################
def test_benchmark_report_serializer_search_sort_pagination_and_delete(monkeypatch) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)

    now = datetime.now(timezone.utc)
    datasets = ["custom/alpha", "custom/beta"]
    with Session(bind=engine) as session:
        for dataset_name in datasets:
            session.add(Dataset(name=dataset_name, status="ready", created_at=now, updated_at=now, ready_at=now))
        session.commit()

    serializer = BenchmarkReportSerializer()
    older = _build_payload("custom/alpha")
    older.update({"run_name": "alpha older", "created_at": "2026-01-01T00:00:00Z"})
    middle = _build_payload("custom/beta")
    middle.update({"run_name": "beta middle", "created_at": "2026-01-02T00:00:00Z"})
    newer = _build_payload("custom/alpha")
    newer.update({"run_name": "alpha newer", "created_at": "2026-01-03T00:00:00Z"})
    older_id = serializer.save_benchmark_report(older)
    middle_id = serializer.save_benchmark_report(middle)
    newer_id = serializer.save_benchmark_report(newer)

    newest = serializer.list_benchmark_reports(BenchmarkReportQuery(limit=2))
    assert [item.report_id for item in newest.reports] == [newer_id, middle_id]
    assert newest.total == 3
    assert newest.offset == 0
    assert newest.limit == 2

    oldest = serializer.list_benchmark_reports(BenchmarkReportQuery(sort="oldest", offset=1, limit=1))
    assert [item.report_id for item in oldest.reports] == [middle_id]
    assert oldest.total == 3

    run_search = serializer.list_benchmark_reports(BenchmarkReportQuery(search="newer"))
    assert [item.report_id for item in run_search.reports] == [newer_id]
    dataset_search = serializer.list_benchmark_reports(BenchmarkReportQuery(search="beta"))
    assert [item.report_id for item in dataset_search.reports] == [middle_id]

    assert serializer.delete_benchmark_report(middle_id) is True
    assert serializer.delete_benchmark_report(middle_id) is False
    assert serializer.load_benchmark_report_by_id(middle_id) is None
    remaining = serializer.list_benchmark_reports(BenchmarkReportQuery(limit=10))
    assert remaining.total == 2
    assert [item.report_id for item in remaining.reports] == [newer_id, older_id]

###############################################################################
def test_benchmark_report_serializer_rejects_v4_payload() -> None:
    serializer = BenchmarkReportSerializer()
    with pytest.raises(ValueError, match="incompatible report version"):
        serializer._normalize_report_row({
            "id": 3,
            "report_version": 4,
            "created_at": datetime.now(timezone.utc),
            "run_name": "old",
            "selected_metric_keys": [],
            "dataset_name": "custom/old",
            "payload": {"schema_version": 3, "methodology_version": "semantic_honesty", "dataset_name": "custom/old"},
        })

###############################################################################
def test_benchmark_report_serializer_rejects_json_encoded_storage() -> None:
    serializer = BenchmarkReportSerializer()

    with pytest.raises(ValueError, match="native JSON object"):
        serializer._normalize_report_row({
            "id": 3,
            "report_version": 5,
            "created_at": datetime.now(timezone.utc),
            "run_name": "encoded",
            "selected_metric_keys": [],
            "payload": '{"schema_version": 3}',
        })

    with pytest.raises(ValueError, match="native JSON array"):
        serializer._normalize_report_row({
            "id": 3,
            "report_version": 5,
            "created_at": datetime.now(timezone.utc),
            "run_name": "encoded",
            "selected_metric_keys": '["metric"]',
            "payload": {"schema_version": 3, "methodology_version": "semantic_honesty", "report_version": 5},
        })
