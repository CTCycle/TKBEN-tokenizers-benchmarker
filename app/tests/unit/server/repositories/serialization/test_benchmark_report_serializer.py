from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from TKBEN.server.repositories.database.backend import get_database
from TKBEN.server.repositories.schemas.models import Base, Dataset
from TKBEN.server.repositories.serialization.benchmark_reports import (
    BenchmarkReportSerializer,
)


def _build_payload(dataset_name: str) -> dict:
    return {
        "status": "success",
        "report_version": 2,
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
        "chart_data": {
            "efficiency": [],
            "fidelity": [],
            "vocabulary": [],
            "fragmentation": [],
            "latency_or_memory_distribution": [],
        },
        "per_document_stats": [],
    }


def test_benchmark_report_serializer_round_trip(monkeypatch) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)

    dataset_name = "custom/serializer_ds"
    with Session(bind=engine) as session:
        session.add(Dataset(name=dataset_name))
        session.commit()

    serializer = BenchmarkReportSerializer()
    payload = _build_payload(dataset_name)

    report_id = serializer.save_benchmark_report(payload)
    stored = serializer.load_benchmark_report_by_id(report_id)
    summaries = serializer.list_benchmark_reports(limit=10)

    assert stored is not None
    assert stored["report_id"] == report_id
    assert stored["dataset_name"] == dataset_name
    assert stored["tokenizers_count"] == 1
    assert summaries[0]["report_id"] == report_id
    assert summaries[0]["dataset_name"] == dataset_name
