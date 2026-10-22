# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from sqlalchemy import create_engine
from datetime import datetime
from datetime import timezone
from sqlalchemy.orm import Session
import pytest
from server.contracts.benchmarks import BenchmarkReportQuery
from server.repositories.database.backend import get_database
from server.repositories.schemas.models import Base
from server.repositories.schemas.models import BenchmarkReport
from server.repositories.schemas.models import Dataset
from server.services.benchmark_reports import BenchmarkReportService
import matplotlib.pyplot as plt
from server.contracts.benchmarks import BenchmarkVisualizationKind
from server.services.benchmark_result_builder import BenchmarkResultBuilder
from server.services.export import DashboardExportService
from server.common.benchmark_metric_definitions import BENCHMARK_METRIC_DEFINITIONS
from server.services.dashboard_export_helpers import DashboardExportFormatting

###############################################################################


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
def test_benchmark_report_service_round_trip(monkeypatch) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)

    dataset_name = "custom/serializer_ds"
    with Session(bind=engine) as session:
        now = datetime.now(timezone.utc)
        session.add(
            Dataset(
                name=dataset_name,
                status="ready",
                created_at=now,
                updated_at=now,
                ready_at=now,
            )
        )
        session.commit()

    report_service = BenchmarkReportService()
    payload = _build_payload(dataset_name)

    report_id = report_service.save_benchmark_report(payload)
    stored = report_service.load_benchmark_report_by_id(report_id)
    summaries = report_service.list_benchmark_reports(BenchmarkReportQuery(limit=10))

    assert stored is not None
    assert stored["report_id"] == report_id
    assert stored["dataset_name"] == dataset_name
    assert stored["tokenizers_count"] == 1
    assert stored["tags"] == []
    assert stored["tokenizer_results"][0]["status"] == "success"
    assert (
        stored["tokenizer_results"][0]["efficiency"]["encode_only_wall_time_seconds"]
        == 1.0
    )
    with Session(bind=engine) as session:
        stored_row = session.get(BenchmarkReport, report_id)
        assert stored_row is not None
        assert "status" not in stored_row.payload
        assert "tags" not in stored_row.payload
        assert "config" in stored_row.payload
    assert summaries.reports[0].report_id == report_id
    assert summaries.reports[0].dataset_name == dataset_name
    assert summaries.reports[0].tags == []
    assert summaries.total == 1

    inconsistent = _build_payload(dataset_name)
    inconsistent["tokenizers_count"] = 2
    with pytest.raises(ValueError, match="tokenizer count disagrees"):
        report_service.save_benchmark_report(inconsistent)

###############################################################################
def test_benchmark_report_service_updates_and_reloads_tags(monkeypatch) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)

    dataset_name = "custom/tagged"
    now = datetime.now(timezone.utc)
    with Session(bind=engine) as session:
        session.add(
            Dataset(
                name=dataset_name,
                status="ready",
                created_at=now,
                updated_at=now,
                ready_at=now,
            )
        )
        session.commit()

    service = BenchmarkReportService()
    report_id = service.save_benchmark_report(_build_payload(dataset_name))

    updated = service.update_benchmark_report_tags(
        report_id, [" Production ", "cpu", "production", ""]
    )

    assert updated is not None
    assert updated.report_id == report_id
    assert updated.tags == ["Production", "cpu"]
    loaded = service.load_benchmark_report_by_id(report_id)
    assert loaded is not None
    assert loaded["tags"] == ["Production", "cpu"]
    summary = service.list_benchmark_reports(BenchmarkReportQuery(limit=10))
    assert summary.reports[0].tags == ["Production", "cpu"]

    with Session(bind=engine) as session:
        stored_row = session.get(BenchmarkReport, report_id)
        assert stored_row is not None
        assert stored_row.tags == ["Production", "cpu"]
        assert "tags" not in stored_row.payload

    assert service.update_benchmark_report_tags(999, ["missing"]) is None

###############################################################################
def test_benchmark_report_service_search_sort_pagination_and_delete(
    monkeypatch,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)

    now = datetime.now(timezone.utc)
    datasets = ["custom/alpha", "custom/beta"]
    with Session(bind=engine) as session:
        for dataset_name in datasets:
            session.add(
                Dataset(
                    name=dataset_name,
                    status="ready",
                    created_at=now,
                    updated_at=now,
                    ready_at=now,
                )
            )
        session.commit()

    report_service = BenchmarkReportService()
    older = _build_payload("custom/alpha")
    older.update({"run_name": "alpha older", "created_at": "2026-01-01T00:00:00Z"})
    middle = _build_payload("custom/beta")
    middle.update({"run_name": "beta middle", "created_at": "2026-01-02T00:00:00Z"})
    newer = _build_payload("custom/alpha")
    newer.update({"run_name": "alpha newer", "created_at": "2026-01-03T00:00:00Z"})
    older_id = report_service.save_benchmark_report(older)
    middle_id = report_service.save_benchmark_report(middle)
    newer_id = report_service.save_benchmark_report(newer)

    newest = report_service.list_benchmark_reports(BenchmarkReportQuery(limit=2))
    assert [item.report_id for item in newest.reports] == [newer_id, middle_id]
    assert newest.total == 3
    assert newest.offset == 0
    assert newest.limit == 2

    oldest = report_service.list_benchmark_reports(
        BenchmarkReportQuery(sort="oldest", offset=1, limit=1)
    )
    assert [item.report_id for item in oldest.reports] == [middle_id]
    assert oldest.total == 3

    run_search = report_service.list_benchmark_reports(
        BenchmarkReportQuery(search="newer")
    )
    assert [item.report_id for item in run_search.reports] == [newer_id]
    dataset_search = report_service.list_benchmark_reports(
        BenchmarkReportQuery(search="beta")
    )
    assert [item.report_id for item in dataset_search.reports] == [middle_id]

    assert report_service.delete_benchmark_report(middle_id) is True
    assert report_service.delete_benchmark_report(middle_id) is False
    assert report_service.load_benchmark_report_by_id(middle_id) is None
    remaining = report_service.list_benchmark_reports(BenchmarkReportQuery(limit=10))
    assert remaining.total == 2
    assert [item.report_id for item in remaining.reports] == [newer_id, older_id]

###############################################################################
def test_benchmark_report_service_rejects_v4_payload() -> None:
    report_service = BenchmarkReportService()
    with pytest.raises(ValueError, match="incompatible report version"):
        report_service._normalize_report_row(
            {
                "id": 3,
                "report_version": 4,
                "created_at": datetime.now(timezone.utc),
                "run_name": "old",
                "selected_metric_keys": [],
                "dataset_name": "custom/old",
                "payload": {
                    "schema_version": 3,
                    "methodology_version": "semantic_honesty",
                    "dataset_name": "custom/old",
                },
            }
        )

###############################################################################
def test_benchmark_report_service_rejects_json_encoded_storage() -> None:
    report_service = BenchmarkReportService()

    with pytest.raises(ValueError, match="native JSON object"):
        report_service._normalize_report_row(
            {
                "id": 3,
                "report_version": 5,
                "created_at": datetime.now(timezone.utc),
                "run_name": "encoded",
                "selected_metric_keys": [],
                "payload": '{"schema_version": 3}',
            }
        )

    with pytest.raises(ValueError, match="native JSON array"):
        report_service._normalize_report_row(
            {
                "id": 3,
                "report_version": 5,
                "created_at": datetime.now(timezone.utc),
                "run_name": "encoded",
                "selected_metric_keys": '["metric"]',
                "payload": {
                    "schema_version": 3,
                    "methodology_version": "semantic_honesty",
                    "report_version": 5,
                },
            }
        )

###############################################################################


###############################################################################
def _widget(
    visualization: str,
    *,
    compatible: list[str],
    points: list[dict] | None = None,
    distributions: list[dict] | None = None,
    buckets: list[dict] | None = None,
) -> dict:
    return {
        "widget_id": f"benchmark.{visualization}",
        "label": visualization,
        "description": visualization,
        "unit": "value",
        "default_visualization": compatible[0],
        "compatible_visualizations": compatible,
        "width": "wide",
        "points": points or [],
        "distributions": distributions or [],
        "buckets": buckets or [],
    }

###############################################################################
def test_metric_definitions_expose_only_strict_visualization_pairs() -> None:
    allowed = {item.value for item in BenchmarkVisualizationKind}
    expected_pairs = {
        "bar": ("bar", "horizontal_bar"),
        "interval_bar": ("interval_bar", "dot_whisker"),
        "box_plot": ("box_plot", "histogram"),
        "grouped_bar": ("grouped_bar", "heatmap"),
    }
    for definition in BENCHMARK_METRIC_DEFINITIONS:
        assert definition.default_visualization.value in allowed
        assert (
            tuple(item.value for item in definition.compatible_visualizations)
            in expected_pairs.values()
        )
        assert (
            definition.compatible_visualizations[0] is definition.default_visualization
        )

###############################################################################
def test_builder_emits_payload_shape_compatible_visualization_choices() -> None:
    result = {
        "tokenizer": "alpha",
        "status": "success",
        "vocabulary_size": 10,
        "fragmentation": {
            "fragmentation_by_word_length_bucket": [
                {"bucket": "short_1_4", "pieces_per_word_mean": 1.2}
            ]
        },
    }
    from server.contracts.benchmarks import BenchmarkTokenizerResult

    dashboard = BenchmarkResultBuilder(None).build_dashboard_data(
        [BenchmarkTokenizerResult.model_validate(result)], {}
    )
    scalar = next(
        widget
        for widget in dashboard.widgets
        if widget.widget_id == "benchmark.meta.vocabulary_size"
    )
    bucket = next(
        widget
        for widget in dashboard.widgets
        if widget.widget_id == "benchmark.frag.fragmentation_by_word_length_bucket"
    )
    assert scalar.default_visualization == BenchmarkVisualizationKind.BAR
    assert scalar.compatible_visualizations == [
        BenchmarkVisualizationKind.BAR,
        BenchmarkVisualizationKind.HORIZONTAL_BAR,
    ]
    assert bucket.default_visualization == BenchmarkVisualizationKind.GROUPED_BAR
    assert bucket.compatible_visualizations == [
        BenchmarkVisualizationKind.GROUPED_BAR,
        BenchmarkVisualizationKind.HEATMAP,
    ]

###############################################################################
def test_builder_preserves_definition_width_for_dense_visualizations() -> None:
    from server.contracts.benchmarks import BenchmarkTokenizerResult

    result = BenchmarkTokenizerResult.model_validate(
        {
            "tokenizer": "alpha",
            "status": "success",
            "vocabulary_size": 10,
            "latency": {"encode_latency_p50_ms": 1.0},
            "fragmentation": {
                "fragmentation_by_word_length_bucket": [
                    {"bucket": "short_1_4", "pieces_per_word_mean": 1.2}
                ]
            },
        }
    )
    dashboard = BenchmarkResultBuilder(None).build_dashboard_data(
        [result],
        {"alpha": [{"elapsed_ns": 1_000_000, "documents": 1}]},
        selected_metric_keys=[
            "lat.encode_latency_distribution",
            "frag.fragmentation_by_word_length_bucket",
        ],
    )
    assert (
        next(
            widget
            for widget in dashboard.widgets
            if widget.widget_id == "benchmark.lat.encode_latency_distribution"
        ).width
        == "wide"
    )
    assert (
        next(
            widget
            for widget in dashboard.widgets
            if widget.widget_id == "benchmark.frag.fragmentation_by_word_length_bucket"
        ).width
        == "wide"
    )

###############################################################################
def test_histogram_bins_use_shared_capped_edges_and_constant_padding() -> None:
    builder = BenchmarkResultBuilder(None)
    bins = builder._histogram_bins(
        {"alpha": list(range(30)), "beta": list(range(10, 40))}
    )
    assert len({(item.bin_low, item.bin_high) for item in bins}) <= 24
    assert {item.tokenizer for item in bins} == {"alpha", "beta"}
    assert all(item.count >= 0 and 0.0 <= item.proportion <= 1.0 for item in bins)

    constant = builder._histogram_bins({"alpha": [3.0, 3.0, 3.0]})
    assert len(constant) == 1
    assert constant[0].bin_low < 3.0 < constant[0].bin_high
    assert constant[0].count == 3
    assert constant[0].proportion == 1.0

###############################################################################
def test_pdf_renderer_covers_all_canonical_visualizations() -> None:
    service = DashboardExportService()
    points = [
        {
            "tokenizer": "alpha",
            "value": 10.0,
            "interval_low": 8.0,
            "interval_high": 12.0,
        },
        {"tokenizer": "beta", "value": 7.0, "interval_low": 6.0, "interval_high": 9.0},
    ]
    distributions = [
        {
            "tokenizer": "alpha",
            "min": 1.0,
            "q1": 2.0,
            "median": 3.0,
            "q3": 4.0,
            "max": 6.0,
            "sample_count": 10,
        }
    ]
    buckets = [
        {"tokenizer": "alpha", "bucket": "short", "value": 1.0},
        {"tokenizer": "alpha", "bucket": "long", "value": 2.0},
        {"tokenizer": "beta", "bucket": "short", "value": 1.5},
        {"tokenizer": "beta", "bucket": "long", "value": 2.5},
    ]
    histogram_bins = [
        {
            "tokenizer": "alpha",
            "bin_low": 0.0,
            "bin_high": 1.0,
            "count": 3,
            "proportion": 1.0,
        }
    ]
    cases = [
        ("bar", ["bar", "horizontal_bar"], points, [], [], []),
        ("horizontal_bar", ["horizontal_bar"], points, [], [], []),
        ("interval_bar", ["interval_bar"], points, [], [], []),
        ("dot_whisker", ["dot_whisker"], points, [], [], []),
        ("box_plot", ["box_plot"], [], distributions, [], []),
        ("histogram", ["histogram"], [], distributions, [], histogram_bins),
        ("grouped_bar", ["grouped_bar"], [], [], buckets, []),
        ("heatmap", ["heatmap"], [], [], buckets, []),
    ]
    for (
        visualization,
        compatible,
        case_points,
        case_distributions,
        case_buckets,
        case_histogram_bins,
    ) in cases:
        figure, axis = plt.subplots()
        try:
            service._render_normalized_benchmark_widget(
                axis,
                _widget(
                    visualization,
                    compatible=compatible,
                    points=case_points,
                    distributions=case_distributions,
                    buckets=case_buckets,
                )
                | {
                    "histogram_bins": case_histogram_bins,
                    "visualization": visualization,
                },
            )
        finally:
            plt.close(figure)

###############################################################################
def test_pdf_benchmark_chart_labels_shorten_long_tokenizer_names() -> None:
    service = DashboardExportService()
    tokenizer_name = "CUSTOM_t3_05_export_validation_20260923"
    expected_label = "CUSTOM_...260923"

    for visualization, compatible, label_axis in (
        ("bar", ["bar"], "x"),
        ("horizontal_bar", ["horizontal_bar"], "y"),
    ):
        figure, axis = plt.subplots()
        try:
            service._render_normalized_benchmark_widget(
                axis,
                _widget(
                    visualization,
                    compatible=compatible,
                    points=[{"tokenizer": tokenizer_name, "value": 23.0}],
                )
                | {"visualization": visualization},
            )

            labels = (
                axis.get_xticklabels()
                if label_axis == "x"
                else axis.get_yticklabels()
            )
            assert [label.get_text() for label in labels] == [expected_label]
            if visualization == "horizontal_bar":
                assert axis.get_xlabel() == "value"
                assert axis.get_ylabel() == "Tokenizer"
            else:
                assert axis.get_xlabel() == "Tokenizer"
                assert axis.get_ylabel() == "value"
        finally:
            plt.close(figure)

###############################################################################
def test_pdf_export_rejects_unknown_and_incompatible_overrides() -> None:
    service = DashboardExportService()
    source = {
        "dataset_name": "custom/test",
        "dashboard": {
            "widgets": [
                _widget(
                    "bar",
                    compatible=["bar", "horizontal_bar"],
                    points=[{"tokenizer": "alpha", "value": 1.0}],
                )
            ]
        },
    }
    with pytest.raises(ValueError, match="unknown visualization"):
        service._normalize_benchmark_dashboard_widgets(
            source,
            {
                "visible_widget_ids": ["benchmark.bar"],
                "ordered_widget_ids": ["benchmark.bar"],
                "visualization_by_widget_id": {"benchmark.bar": "bogus"},
            },
        )
    with pytest.raises(ValueError, match="incompatible"):
        service._normalize_benchmark_dashboard_widgets(
            source,
            {
                "visible_widget_ids": ["benchmark.bar"],
                "ordered_widget_ids": ["benchmark.bar"],
                "visualization_by_widget_id": {"benchmark.bar": "heatmap"},
            },
        )

###############################################################################


###############################################################################
def build_dataset_payload() -> dict:
    return {
        "report": {
            "dataset_name": "wikitext/wikitext-2-v1",
            "report_id": 12,
            "created_at": "2026-03-04T12:00:00Z",
            "document_count": 5000,
            "document_length_histogram": {
                "bins": ["0-10", "10-20", "20-30", "30-40"],
                "counts": [12, 84, 66, 18],
            },
            "word_length_histogram": {
                "bins": ["1", "2", "3", "4", "5+"],
                "counts": [25, 47, 61, 39, 22],
            },
            "most_common_words": [
                {"word": "the", "count": 250},
                {"word": "of", "count": 210},
                {"word": "and", "count": 196},
                {"word": "to", "count": 182},
            ],
            "aggregate_statistics": {
                "doc.length_mean": 74.82,
                "doc.length_min": 1,
                "doc.length_max": 510,
                "doc.length_cv": 0.43,
                "doc.length_p50": 68,
                "doc.length_p90": 132,
                "doc.length_p99": 228,
                "corpus.unique_words": 15230,
                "corpus.mattr": 0.8312,
                "words.shannon_entropy": 7.2345,
                "words.hapax_ratio": 0.5123,
                "words.zipf_slope": -1.0211,
                "words.frequency_gini": 0.7563,
                "words.hhi": 0.001421,
                "chars.whitespace_ratio": 0.21,
                "chars.punctuation_ratio": 0.09,
                "chars.digit_ratio": 0.02,
                "chars.uppercase_ratio": 0.05,
                "chars.non_ascii_ratio": 0.01,
                "chars.control_ratio": 0.0,
                "chars.other_ratio": 0.62,
                "quality.duplicate_rate": 0.031,
                "quality.near_duplicate_rate": 0.072,
                "words.topk_concentration": 0.412,
                "words.rare_tail_mass": 0.133,
                "words.normalized_entropy": 0.74,
                "words.zipf_curve": [
                    {"rank": 1, "frequency": 300},
                    {"rank": 2, "frequency": 205},
                    {"rank": 3, "frequency": 150},
                    {"rank": 4, "frequency": 110},
                ],
            },
        }
    }

###############################################################################
def build_tokenizer_payload() -> dict:
    return {
        "report": {
            "report_id": 22,
            "tokenizer_name": "bert-base-uncased",
            "vocabulary_size": 30522,
            "huggingface_url": "https://huggingface.co/bert-base-uncased",
            "global_stats": {
                "tokenizer_class": "BertTokenizerFast",
                "base_vocabulary_size": 30522,
                "model_max_length": 512,
                "padding_side": "right",
                "special_tokens_count": 5,
                "added_tokens_count": 0,
            },
            "token_length_histogram": {
                "bins": ["1", "2", "3", "4", "5", "6+"],
                "counts": [1220, 8045, 9102, 6703, 3987, 1465],
            },
        },
        "vocabulary_items": [
            {"token_id": 0, "token": "[PAD]", "length": 5},
            {"token_id": 1, "token": "[UNK]", "length": 5},
            {"token_id": 2, "token": "[CLS]", "length": 5},
            {"token_id": 3, "token": "[SEP]", "length": 5},
            {"token_id": 4, "token": "[MASK]", "length": 6},
            {"token_id": 5, "token": "the", "length": 3},
        ],
    }

###############################################################################
def build_benchmark_payload() -> dict:
    return {
        "report": {
            "report_id": 8,
            "dataset_name": "wikitext/wikitext-2-v1",
            "documents_processed": 3000,
            "tokenizers_count": 3,
            "dashboard": {
                "widgets": [
                    {
                        "widget_id": "benchmark.vocabulary",
                        "label": "Vocabulary size",
                        "description": "Vocabulary size by tokenizer",
                        "unit": "tokens",
                        "default_visualization": "bar",
                        "compatible_visualizations": ["bar", "horizontal_bar"],
                        "width": "standard",
                        "points": [
                            {"tokenizer": "bert-base-uncased", "value": 30522},
                            {"tokenizer": "gpt2", "value": 50257},
                        ],
                    },
                    {
                        "widget_id": "benchmark.speed",
                        "label": "Tokens per second",
                        "description": "Tokenization throughput",
                        "unit": "tokens/s",
                        "default_visualization": "interval_bar",
                        "compatible_visualizations": ["interval_bar", "dot_whisker"],
                        "width": "wide",
                        "points": [
                            {
                                "tokenizer": "bert-base-uncased",
                                "value": 11000,
                                "interval_low": 10500,
                                "interval_high": 11500,
                            },
                            {
                                "tokenizer": "gpt2",
                                "value": 9600,
                                "interval_low": 9200,
                                "interval_high": 10000,
                            },
                        ],
                    },
                    {
                        "widget_id": "benchmark.hidden",
                        "label": "Hidden metric",
                        "description": "Must not be exported",
                        "unit": "ratio",
                        "default_visualization": "bar",
                        "compatible_visualizations": ["bar", "horizontal_bar"],
                        "width": "standard",
                        "points": [{"tokenizer": "bert-base-uncased", "value": 1}],
                    },
                ],
            },
        },
        "selected_distribution_tokenizer": "bert-base-uncased",
        "visible_widget_ids": ["benchmark.speed", "benchmark.vocabulary"],
        "ordered_widget_ids": [
            "benchmark.speed",
            "benchmark.hidden",
            "benchmark.vocabulary",
        ],
        "visualization_by_widget_id": {
            "benchmark.speed": "dot_whisker",
            "benchmark.vocabulary": "horizontal_bar",
        },
    }

###############################################################################
def test_export_dataset_dashboard_pdf_generates_pdf_bytes() -> None:
    service = DashboardExportService()
    result = service.export_dashboard_pdf(
        dashboard_type="dataset",
        report_name="dataset-layout-export",
        file_name="dataset-layout-export",
        dashboard_payload=build_dataset_payload(),
    )

    assert result.file_name == "dataset-layout-export.pdf"
    assert result.page_count >= 2
    assert result.pdf_bytes.startswith(b"%PDF")

###############################################################################
def test_export_tokenizer_dashboard_pdf_generates_pdf_bytes() -> None:
    service = DashboardExportService()
    result = service.export_dashboard_pdf(
        dashboard_type="tokenizer",
        report_name="tokenizer-layout-export",
        file_name="tokenizer-layout-export",
        dashboard_payload=build_tokenizer_payload(),
    )

    assert result.file_name == "tokenizer-layout-export.pdf"
    assert result.page_count >= 1
    assert result.pdf_bytes.startswith(b"%PDF")

###############################################################################
def test_export_benchmark_dashboard_pdf_generates_pdf_bytes() -> None:
    service = DashboardExportService()
    result = service.export_dashboard_pdf(
        dashboard_type="benchmark",
        report_name="benchmark-layout-export",
        file_name="benchmark-layout-export",
        dashboard_payload=build_benchmark_payload(),
    )

    assert result.file_name == "benchmark-layout-export.pdf"
    assert result.page_count == 2
    assert result.pdf_bytes.startswith(b"%PDF")

###############################################################################
def test_benchmark_pdf_normalizes_visible_widgets_in_requested_order() -> None:
    service = DashboardExportService()
    payload = build_benchmark_payload()
    source = payload["report"]

    widgets = service._normalize_benchmark_dashboard_widgets(source, payload)

    assert [widget["widget_id"] for widget in widgets] == [
        "benchmark.speed",
        "benchmark.vocabulary",
    ]

###############################################################################
def test_export_dashboard_pdf_rejects_unsupported_dashboard_type() -> None:
    service = DashboardExportService()
    with pytest.raises(ValueError, match="Unsupported dashboard type"):
        service.export_dashboard_pdf(
            dashboard_type="invalid",
            report_name="x",
            file_name="x",
            dashboard_payload={},
        )

###############################################################################
def test_dashboard_export_formatting_accepts_only_current_metric_shapes() -> None:
    formatting = DashboardExportFormatting()

    assert formatting._parse_zipf_curve(
        [
            {"rank": 1, "frequency": 10},
        ]
    ) == [{"rank": 1.0, "frequency": 10.0}]
    assert formatting._parse_zipf_curve([[1, 10]]) == []
    assert formatting._parse_zipf_curve('[{"rank": 1, "frequency": 10}]') == []
    assert formatting._parse_word_frequency(
        [
            {"word": "hello", "count": 3},
        ]
    ) == [{"word": "hello", "count": 3}]
    assert (
        formatting._parse_word_frequency(
            [
                {"token": "hello", "count": 3},
            ]
        )
        == []
    )
