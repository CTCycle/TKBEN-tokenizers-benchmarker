from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from server.common.utils.security import contains_control_chars, normalize_identifier

###############################################################################
class BenchmarkVisualizationKind(StrEnum):
    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"
    INTERVAL_BAR = "interval_bar"
    DOT_WHISKER = "dot_whisker"
    BOX_PLOT = "box_plot"
    HISTOGRAM = "histogram"
    GROUPED_BAR = "grouped_bar"
    HEATMAP = "heatmap"

###############################################################################
class BenchmarkRunConfig(BaseModel):
    max_documents: int = Field(default=0, ge=0)
    warmup_trials: int = Field(default=2, ge=0, le=100)
    timed_trials: int = Field(default=8, ge=1, le=200)
    batch_size: int = Field(default=16, ge=1, le=4096)
    seed: int = Field(default=42)
    parallelism: int = Field(default=1, ge=1, le=128)
    include_lm_metrics: bool = Field(default=False)
    add_special_tokens: bool = Field(default=False)
    padding: bool = Field(default=False)
    truncation: bool = Field(default=False)
    max_length: int | None = Field(default=None, ge=1)
    store_per_document_stats: bool = Field(default=True)
    per_document_sample_size: int = Field(default=500, ge=1, le=10000)

###############################################################################
class BenchmarkHardwareProfile(BaseModel):
    runtime: str = Field(default="")
    os: str = Field(default="")
    cpu_model: str | None = Field(default=None)
    cpu_logical_cores: int | None = Field(default=None)
    memory_total_mb: float | None = Field(default=None)

###############################################################################
class BenchmarkTrialSummary(BaseModel):
    warmup_trials: int = Field(default=0)
    timed_trials: int = Field(default=0)

###############################################################################
class BenchmarkEfficiencyMetrics(BaseModel):
    encode_tokens_per_second_mean: float | None = None
    encode_tokens_per_second_ci95_low: float | None = None
    encode_tokens_per_second_ci95_high: float | None = None
    encode_chars_per_second_mean: float | None = None
    encode_bytes_per_second_mean: float | None = None
    encode_only_wall_time_seconds: float | None = None
    dataset_stream_wall_time_seconds: float | None = None
    postprocess_wall_time_seconds: float | None = None
    end_to_end_wall_time_seconds: float | None = None
    load_time_seconds: float | None = None

###############################################################################
class BenchmarkLatencyMetrics(BaseModel):
    encode_latency_p50_ms: float | None = None
    encode_latency_p95_ms: float | None = None
    encode_latency_p99_ms: float | None = None
    sample_count: int | None = None

###############################################################################
class BenchmarkFidelityMetrics(BaseModel):
    exact_round_trip_rate: float | None = Field(default=None)
    normalized_round_trip_rate: float | None = Field(default=None)
    unknown_token_rate: float | None = Field(default=None)
    byte_fallback_rate: float | None = Field(default=None)
    lossless_encodability_rate: float | None = Field(default=None)

###############################################################################
class BenchmarkFragmentationBucket(BaseModel):
    bucket: str
    pieces_per_word_mean: float = Field(default=0.0)

###############################################################################
class BenchmarkFragmentationMetrics(BaseModel):
    tokens_per_character: float | None = Field(default=None)
    characters_per_token: float | None = Field(default=None)
    tokens_per_byte: float | None = Field(default=None)
    bytes_per_token: float | None = Field(default=None)
    pieces_per_word_mean: float | None = Field(default=None)
    fragmentation_by_word_length_bucket: list[BenchmarkFragmentationBucket] = Field(
        default_factory=list
    )

###############################################################################
class BenchmarkResourceMetrics(BaseModel):
    peak_rss_mb: float | None = Field(default=None)
    memory_delta_mb: float | None = Field(default=None)

###############################################################################
class BenchmarkTokenizerResult(BaseModel):
    tokenizer: str
    status: str = Field(default="success")
    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)
    tokenizer_family: str = Field(default="unknown")
    runtime_backend: str = Field(default="unknown")
    vocabulary_size: int = Field(default=0)
    added_tokens: int = Field(default=0)
    special_token_share: float = Field(default=0.0)
    efficiency: BenchmarkEfficiencyMetrics = Field(
        default_factory=BenchmarkEfficiencyMetrics
    )
    latency: BenchmarkLatencyMetrics = Field(default_factory=BenchmarkLatencyMetrics)
    fidelity: BenchmarkFidelityMetrics = Field(default_factory=BenchmarkFidelityMetrics)
    fragmentation: BenchmarkFragmentationMetrics = Field(
        default_factory=BenchmarkFragmentationMetrics
    )
    resources: BenchmarkResourceMetrics = Field(
        default_factory=BenchmarkResourceMetrics
    )

###############################################################################
class BenchmarkSeriesPoint(BaseModel):
    tokenizer: str
    value: float = Field(default=0.0)
    ci95_low: float | None = Field(default=None)
    ci95_high: float | None = Field(default=None)

###############################################################################
class BenchmarkDistributionPoint(BaseModel):
    tokenizer: str
    min: float = Field(default=0.0)
    q1: float = Field(default=0.0)
    median: float = Field(default=0.0)
    q3: float = Field(default=0.0)
    max: float = Field(default=0.0)
    sample_count: int = Field(default=0)

###############################################################################
class BenchmarkDashboardPoint(BaseModel):
    tokenizer: str
    value: float
    interval_low: float | None = None
    interval_high: float | None = None

###############################################################################
class BenchmarkDashboardDistribution(BaseModel):
    tokenizer: str
    min: float
    q1: float
    median: float
    q3: float
    max: float
    sample_count: int

###############################################################################
class BenchmarkDashboardBucketPoint(BaseModel):
    tokenizer: str
    bucket: str
    value: float

###############################################################################
class BenchmarkDashboardHistogramBin(BaseModel):
    tokenizer: str
    bin_low: float
    bin_high: float
    count: int
    proportion: float

###############################################################################
class BenchmarkDashboardWidgetData(BaseModel):
    widget_id: str
    metric_keys: list[str]
    category_key: str
    category_label: str
    label: str
    description: str
    unit: str
    display_format: str
    default_visualization: BenchmarkVisualizationKind
    compatible_visualizations: list[BenchmarkVisualizationKind]
    default_visible: bool
    width: str
    points: list[BenchmarkDashboardPoint] = Field(default_factory=list)
    distributions: list[BenchmarkDashboardDistribution] = Field(default_factory=list)
    buckets: list[BenchmarkDashboardBucketPoint] = Field(default_factory=list)
    histogram_bins: list[BenchmarkDashboardHistogramBin] = Field(default_factory=list)

###############################################################################
class BenchmarkDashboardData(BaseModel):
    widgets: list[BenchmarkDashboardWidgetData] = Field(default_factory=list)
    available_widget_ids: list[str] = Field(default_factory=list)
    available_metric_keys: list[str] = Field(default_factory=list)
    unavailable_selected_metric_keys: list[str] = Field(default_factory=list)

###############################################################################
class BenchmarkRunRequest(BaseModel):
    tokenizers: list[str] = Field(..., description="List of tokenizer IDs to benchmark")
    dataset_name: str = Field(..., description="Name of the dataset to use")
    config: BenchmarkRunConfig = Field(default_factory=BenchmarkRunConfig)
    custom_tokenizer_name: str | None = Field(default=None)
    run_name: str | None = Field(default=None)
    selected_metric_keys: list[str] | None = Field(default=None)

    # -------------------------------------------------------------------------
    @field_validator("tokenizers")
    @classmethod
    def validate_tokenizers(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for tokenizer in value:
            cleaned = normalize_identifier(
                tokenizer, "Tokenizer identifier", max_length=160
            )
            if cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        if len(normalized) > 200:
            raise ValueError("Too many tokenizers requested (max 200).")
        return normalized

    # -------------------------------------------------------------------------
    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, value: str) -> str:
        if not value.strip():
            return ""
        return normalize_identifier(value, "Dataset name", max_length=200)

    # -------------------------------------------------------------------------
    @field_validator("custom_tokenizer_name")
    @classmethod
    def validate_custom_tokenizer_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_identifier(value, "Custom tokenizer name", max_length=160)

    # -

    # -------------------------------------------------------------------------
    @field_validator("run_name")
    @classmethod
    def validate_run_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if len(normalized) > 120:
            raise ValueError("Run name is too long (max 120 characters).")
        if contains_control_chars(normalized):
            raise ValueError("Run name contains unsupported control characters.")
        return normalized

###############################################################################
class BenchmarkMetricCatalogMetric(BaseModel):
    key: str
    label: str
    description: str
    scope: str
    value_kind: str
    source: str = Field(default="observed")
    core: bool = Field(default=False)
    unit: str = ""
    display_format: str = "number"
    default_visualization: BenchmarkVisualizationKind = BenchmarkVisualizationKind.BAR
    compatible_visualizations: list[BenchmarkVisualizationKind] = Field(default_factory=lambda: [BenchmarkVisualizationKind.BAR, BenchmarkVisualizationKind.HORIZONTAL_BAR])
    default_visible: bool = False
    width: str = "standard"

###############################################################################
class BenchmarkMetricCatalogCategory(BaseModel):
    category_key: str
    category_label: str
    metrics: list[BenchmarkMetricCatalogMetric] = Field(default_factory=list)

###############################################################################
class BenchmarkMetricCatalogResponse(BaseModel):
    categories: list[BenchmarkMetricCatalogCategory] = Field(default_factory=list)

###############################################################################
class BenchmarkReportSummary(BaseModel):
    report_id: int
    report_version: int
    created_at: str | None = Field(default=None)
    run_name: str | None = Field(default=None)
    dataset_name: str
    documents_processed: int = Field(default=0)
    tokenizers_count: int = Field(default=0)
    tokenizers_processed: list[str] = Field(default_factory=list)
    selected_metric_keys: list[str] = Field(default_factory=list)

###############################################################################
class BenchmarkReportListResponse(BaseModel):
    reports: list[BenchmarkReportSummary] = Field(default_factory=list)

###############################################################################
class BenchmarkPerDocumentTokenizerStats(BaseModel):
    tokenizer: str
    tokens_count: list[int] = Field(default_factory=list)
    bytes_per_token: list[float] = Field(default_factory=list)
    pieces_per_word: list[float | None] = Field(default_factory=list)
    encode_latency_ms: list[float | None] = Field(default_factory=list)
    peak_rss_mb: list[float | None] = Field(default_factory=list)

###############################################################################
class BenchmarkRunResponse(BaseModel):
    status: str = Field(default="success")
    schema_version: int = Field(default=3)
    methodology_version: str = Field(default="semantic_honesty")
    report_id: int | None = Field(default=None)
    report_version: int = Field(default=5)
    created_at: str | None = Field(default=None)
    run_name: str | None = Field(default=None)
    selected_metric_keys: list[str] = Field(default_factory=list)
    dataset_name: str
    documents_processed: int
    tokenizers_processed: list[str] = Field(default_factory=list)
    tokenizers_count: int = Field(default=0)
    config: BenchmarkRunConfig = Field(default_factory=BenchmarkRunConfig)
    hardware_profile: BenchmarkHardwareProfile = Field(
        default_factory=BenchmarkHardwareProfile
    )
    trial_summary: BenchmarkTrialSummary = Field(default_factory=BenchmarkTrialSummary)
    tokenizer_results: list[BenchmarkTokenizerResult] = Field(default_factory=list)
    dashboard: BenchmarkDashboardData = Field(default_factory=BenchmarkDashboardData)
    per_document_stats: list[BenchmarkPerDocumentTokenizerStats] = Field(
        default_factory=list
    )
    runtime_metadata: dict[str, object] = Field(default_factory=dict)
    raw_observations: dict[str, list[dict[str, object]]] = Field(default_factory=dict)
