"""Canonical, transport-neutral benchmark metric and dashboard definitions."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from server.domain.benchmarks import BenchmarkVisualizationKind

###############################################################################
class BenchmarkWidgetWidth(StrEnum):
    STANDARD = "standard"
    WIDE = "wide"

###############################################################################
@dataclass(frozen=True)
class BenchmarkMetricDefinition:
    key: str
    widget_id: str
    category_key: str
    category_label: str
    label: str
    description: str
    path: str
    unit: str
    display_format: str
    default_visualization: BenchmarkVisualizationKind = BenchmarkVisualizationKind.BAR
    compatible_visualizations: tuple[BenchmarkVisualizationKind, ...] = (BenchmarkVisualizationKind.BAR, BenchmarkVisualizationKind.LOLLIPOP)
    width: BenchmarkWidgetWidth = BenchmarkWidgetWidth.STANDARD
    default_visible: bool = False
    required_metric_keys: tuple[str, ...] = ()
    interval_low_path: str | None = None
    interval_high_path: str | None = None
    distribution_source: str | None = None

###############################################################################
def _definition(key: str, category_key: str, category_label: str, label: str, path: str, unit: str, display_format: str, *, default_visible: bool = False, default_visualization: BenchmarkVisualizationKind = BenchmarkVisualizationKind.BAR, compatible_visualizations: tuple[BenchmarkVisualizationKind, ...] | None = None, width: BenchmarkWidgetWidth = BenchmarkWidgetWidth.STANDARD, interval_low_path: str | None = None, interval_high_path: str | None = None, distribution_source: str | None = None, required_metric_keys: tuple[str, ...] = ()) -> BenchmarkMetricDefinition:
    return BenchmarkMetricDefinition(key=key, widget_id=f"benchmark.{key}", category_key=category_key, category_label=category_label, label=label, description=label, path=path, unit=unit, display_format=display_format, default_visualization=default_visualization, compatible_visualizations=compatible_visualizations or (default_visualization, BenchmarkVisualizationKind.LOLLIPOP if default_visualization is BenchmarkVisualizationKind.BAR else default_visualization), width=width, interval_low_path=interval_low_path, interval_high_path=interval_high_path, distribution_source=distribution_source, required_metric_keys=required_metric_keys or (key,))


BENCHMARK_METRIC_DEFINITIONS: tuple[BenchmarkMetricDefinition, ...] = (
    _definition("meta.vocabulary_size", "metadata", "Tokenizer metadata", "Vocabulary size", "vocabulary_size", "tokens", "count", default_visible=True),
    _definition("meta.added_tokens", "metadata", "Tokenizer metadata", "Added token count", "added_tokens", "tokens", "count"),
    _definition("meta.special_token_share", "metadata", "Tokenizer metadata", "Special-token share", "special_token_share", "%", "percent"),
    _definition("eff.encode_tokens_per_second_mean", "efficiency", "Efficiency", "Tokens per second", "efficiency.encode_tokens_per_second_mean", "tokens/s", "throughput", default_visible=True, default_visualization=BenchmarkVisualizationKind.INTERVAL_BAR, compatible_visualizations=(BenchmarkVisualizationKind.INTERVAL_BAR, BenchmarkVisualizationKind.DOT_WHISKER), interval_low_path="efficiency.encode_tokens_per_second_ci95_low", interval_high_path="efficiency.encode_tokens_per_second_ci95_high", required_metric_keys=("eff.encode_tokens_per_second_mean", "eff.encode_tokens_per_second_ci95")),
    _definition("eff.encode_chars_per_second_mean", "efficiency", "Efficiency", "Characters per second", "efficiency.encode_chars_per_second_mean", "chars/s", "throughput", default_visible=True),
    _definition("eff.encode_bytes_per_second_mean", "efficiency", "Efficiency", "Bytes per second", "efficiency.encode_bytes_per_second_mean", "bytes/s", "throughput"),
    *tuple(_definition(f"eff.{key}", "efficiency", "Efficiency", label, f"efficiency.{key}", "s", "seconds") for key, label in (("encode_only_wall_time_seconds", "Encode-only time"), ("dataset_stream_wall_time_seconds", "Dataset stream time"), ("postprocess_wall_time_seconds", "Post-processing time"), ("end_to_end_wall_time_seconds", "End-to-end time"), ("load_time_seconds", "Load time"))),
    *tuple(_definition(f"lat.{key}", "latency", "Latency", label, f"latency.{key}", "ms", "milliseconds", default_visible=key == "encode_latency_p50_ms") for key, label in (("encode_latency_p50_ms", "Latency p50"), ("encode_latency_p95_ms", "Latency p95"), ("encode_latency_p99_ms", "Latency p99"))),
    _definition("lat.sample_count", "latency", "Latency", "Timed observation count", "latency.sample_count", "observations", "count"),
    _definition("lat.encode_latency_distribution", "latency", "Latency", "Timed-observation latency distribution", "latency.encode_latency_p50_ms", "ms", "milliseconds", default_visualization=BenchmarkVisualizationKind.BOX_PLOT, compatible_visualizations=(BenchmarkVisualizationKind.BOX_PLOT, BenchmarkVisualizationKind.RANGE_PLOT), width=BenchmarkWidgetWidth.WIDE, distribution_source="raw_latency"),
    *tuple(_definition(f"fid.{key}", "fidelity", "Fidelity", label, f"fidelity.{key}", "%", "percent", default_visible=key in {"exact_round_trip_rate", "normalized_round_trip_rate"}) for key, label in (("exact_round_trip_rate", "Exact token-ID round trip"), ("normalized_round_trip_rate", "Normalized text round trip"), ("unknown_token_rate", "Unknown-token rate"), ("byte_fallback_rate", "Byte-fallback rate"), ("lossless_encodability_rate", "Vocabulary character overlap"))),
    *tuple(_definition(f"frag.{key}", "fragmentation", "Fragmentation", label, f"fragmentation.{key}", "ratio", "ratio", default_visible=key == "pieces_per_word_mean") for key, label in (("tokens_per_character", "Tokens per character"), ("characters_per_token", "Characters per token"), ("tokens_per_byte", "Tokens per byte"), ("bytes_per_token", "Bytes per token"), ("pieces_per_word_mean", "Pieces per word"))),
    _definition("frag.fragmentation_by_word_length_bucket", "fragmentation", "Fragmentation", "Word-length bucket comparison", "fragmentation.fragmentation_by_word_length_bucket", "pieces/word", "ratio", default_visualization=BenchmarkVisualizationKind.GROUPED_BAR, compatible_visualizations=(BenchmarkVisualizationKind.GROUPED_BAR, BenchmarkVisualizationKind.HEATMAP), width=BenchmarkWidgetWidth.WIDE),
    _definition("res.peak_rss_mb", "resources", "Resources", "Peak RSS", "resources.peak_rss_mb", "MB", "megabytes"),
    _definition("res.memory_delta_mb", "resources", "Resources", "Memory delta", "resources.memory_delta_mb", "MB", "megabytes"),
    *tuple(_definition(key, "per_document", "Per-document", label, "", unit, display, default_visualization=BenchmarkVisualizationKind.BOX_PLOT, compatible_visualizations=(BenchmarkVisualizationKind.BOX_PLOT, BenchmarkVisualizationKind.RANGE_PLOT), width=BenchmarkWidgetWidth.WIDE, distribution_source=source) for key, label, unit, display, source in (("doc.tokens_count_distribution", "Token-count distribution", "tokens", "count", "tokens_count"), ("doc.bytes_per_token_distribution", "Bytes-per-token distribution", "ratio", "ratio", "bytes_per_token"), ("doc.pieces_per_word_distribution", "Pieces-per-word distribution", "ratio", "ratio", "pieces_per_word"), ("doc.encode_latency_distribution", "Encode-latency distribution", "ms", "milliseconds", "encode_latency_ms"), ("doc.peak_rss_distribution", "Peak-RSS distribution", "MB", "megabytes", "peak_rss_mb"))),
)

BENCHMARK_DEFINITION_BY_KEY = {definition.key: definition for definition in BENCHMARK_METRIC_DEFINITIONS}
BENCHMARK_DEFINITION_BY_WIDGET_ID = {definition.widget_id: definition for definition in BENCHMARK_METRIC_DEFINITIONS}

###############################################################################
def benchmark_metric_catalog() -> list[dict[str, object]]:
    categories: dict[str, dict[str, object]] = {}
    for definition in BENCHMARK_METRIC_DEFINITIONS:
        category = categories.setdefault(definition.category_key, {"category_key": definition.category_key, "category_label": definition.category_label, "metrics": []})
        metrics = category["metrics"]
        assert isinstance(metrics, list)
        metrics.append({"key": definition.key, "label": definition.label, "description": definition.description, "scope": "per_document" if definition.distribution_source in {"tokens_count", "bytes_per_token", "pieces_per_word", "encode_latency_ms", "peak_rss_mb"} else "tokenizer_global", "value_kind": definition.default_visualization.value, "core": definition.default_visible, "unit": definition.unit, "display_format": definition.display_format, "default_visualization": definition.default_visualization.value, "compatible_visualizations": [item.value for item in definition.compatible_visualizations], "default_visible": definition.default_visible, "width": definition.width.value})
    return list(categories.values())
