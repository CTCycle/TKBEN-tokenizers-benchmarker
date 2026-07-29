from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from pydantic import ValidationError

from server.domain.benchmarks import BenchmarkDashboardWidgetData, BenchmarkVisualizationKind
from server.services.benchmark_result_builder import BenchmarkResultBuilder
from server.services.export import DashboardExportService
from server.services.metrics.benchmark_definitions import BENCHMARK_METRIC_DEFINITIONS

###############################################################################
def _widget(visualization: str, *, compatible: list[str], points: list[dict] | None = None, distributions: list[dict] | None = None, buckets: list[dict] | None = None) -> dict:
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
        "bar": ("bar", "lollipop"),
        "interval_bar": ("interval_bar", "dot_whisker"),
        "box_plot": ("box_plot", "range_plot"),
        "grouped_bar": ("grouped_bar", "heatmap"),
    }
    for definition in BENCHMARK_METRIC_DEFINITIONS:
        assert definition.default_visualization.value in allowed
        assert tuple(item.value for item in definition.compatible_visualizations) in expected_pairs.values()
        assert definition.compatible_visualizations[0] is definition.default_visualization

###############################################################################
def test_dashboard_model_rejects_unknown_visualization() -> None:
    with pytest.raises(ValidationError):
        BenchmarkDashboardWidgetData(
            widget_id="benchmark.invalid",
            metric_keys=["invalid"],
            category_key="test",
            category_label="Test",
            label="Invalid",
            description="Invalid",
            unit="value",
            display_format="number",
            default_visualization="not_a_chart",
            compatible_visualizations=["not_a_chart"],
            default_visible=True,
            width="standard",
        )

###############################################################################
def test_builder_emits_payload_shape_compatible_visualization_choices() -> None:
    result = {
        "tokenizer": "alpha",
        "status": "success",
        "vocabulary_size": 10,
        "fragmentation": {"fragmentation_by_word_length_bucket": [{"bucket": "short_1_4", "pieces_per_word_mean": 1.2}]},
    }
    from server.domain.benchmarks import BenchmarkTokenizerResult

    dashboard = BenchmarkResultBuilder(None).build_dashboard_data([BenchmarkTokenizerResult.model_validate(result)], {})
    scalar = next(widget for widget in dashboard.widgets if widget.widget_id == "benchmark.meta.vocabulary_size")
    bucket = next(widget for widget in dashboard.widgets if widget.widget_id == "benchmark.frag.fragmentation_by_word_length_bucket")
    assert scalar.default_visualization == BenchmarkVisualizationKind.BAR
    assert scalar.compatible_visualizations == [BenchmarkVisualizationKind.BAR, BenchmarkVisualizationKind.LOLLIPOP]
    assert bucket.default_visualization == BenchmarkVisualizationKind.GROUPED_BAR
    assert bucket.compatible_visualizations == [BenchmarkVisualizationKind.GROUPED_BAR, BenchmarkVisualizationKind.HEATMAP]

###############################################################################
def test_pdf_renderer_covers_all_canonical_visualizations() -> None:
    service = DashboardExportService()
    points = [{"tokenizer": "alpha", "value": 10.0, "interval_low": 8.0, "interval_high": 12.0}, {"tokenizer": "beta", "value": 7.0, "interval_low": 6.0, "interval_high": 9.0}]
    distributions = [{"tokenizer": "alpha", "min": 1.0, "q1": 2.0, "median": 3.0, "q3": 4.0, "max": 6.0, "sample_count": 10}]
    buckets = [{"tokenizer": "alpha", "bucket": "short", "value": 1.0}, {"tokenizer": "alpha", "bucket": "long", "value": 2.0}, {"tokenizer": "beta", "bucket": "short", "value": 1.5}, {"tokenizer": "beta", "bucket": "long", "value": 2.5}]
    cases = [("bar", ["bar", "lollipop"], points, [], []), ("lollipop", ["lollipop"], points, [], []), ("interval_bar", ["interval_bar"], points, [], []), ("dot_whisker", ["dot_whisker"], points, [], []), ("box_plot", ["box_plot"], [], distributions, []), ("range_plot", ["range_plot"], [], distributions, []), ("grouped_bar", ["grouped_bar"], [], [], buckets), ("heatmap", ["heatmap"], [], [], buckets)]
    for visualization, compatible, case_points, case_distributions, case_buckets in cases:
        figure, axis = plt.subplots()
        try:
            service._render_normalized_benchmark_widget(axis, _widget(visualization, compatible=compatible, points=case_points, distributions=case_distributions, buckets=case_buckets) | {"visualization": visualization})
        finally:
            plt.close(figure)

###############################################################################
def test_pdf_export_rejects_unknown_and_incompatible_overrides() -> None:
    service = DashboardExportService()
    source = {"dataset_name": "custom/test", "dashboard": {"widgets": [_widget("bar", compatible=["bar", "lollipop"], points=[{"tokenizer": "alpha", "value": 1.0}])]}}
    with pytest.raises(ValueError, match="unknown visualization"):
        service._normalize_benchmark_dashboard_widgets(source, {"visible_widget_ids": ["benchmark.bar"], "ordered_widget_ids": ["benchmark.bar"], "visualization_by_widget_id": {"benchmark.bar": "bogus"}})
    with pytest.raises(ValueError, match="incompatible"):
        service._normalize_benchmark_dashboard_widgets(source, {"visible_widget_ids": ["benchmark.bar"], "ordered_widget_ids": ["benchmark.bar"], "visualization_by_widget_id": {"benchmark.bar": "heatmap"}})
