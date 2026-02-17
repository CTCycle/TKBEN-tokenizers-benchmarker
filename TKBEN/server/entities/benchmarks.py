from __future__ import annotations

from pydantic import BaseModel, Field


###############################################################################
# Chart data models for frontend visualization
###############################################################################
class VocabularyStats(BaseModel):
    """Vocabulary statistics for a single tokenizer."""
    tokenizer: str
    vocabulary_size: int = Field(default=0)
    subwords_count: int = Field(default=0)
    true_words_count: int = Field(default=0)
    subwords_percentage: float = Field(default=0.0)


class TokenLengthBin(BaseModel):
    """A single bin in token length histogram."""
    bin_start: int
    bin_end: int
    count: int


class TokenLengthDistribution(BaseModel):
    """Token length distribution for a tokenizer."""
    tokenizer: str
    bins: list[TokenLengthBin] = Field(default_factory=list)
    mean: float = Field(default=0.0)
    std: float = Field(default=0.0)


class SpeedMetric(BaseModel):
    """Speed and throughput metrics for comparison charts."""
    tokenizer: str
    tokens_per_second: float = Field(default=0.0)
    chars_per_second: float = Field(default=0.0)
    processing_time_seconds: float = Field(default=0.0)


class ChartData(BaseModel):
    """All chart data for frontend visualization."""
    vocabulary_stats: list[VocabularyStats] = Field(default_factory=list)
    token_length_distributions: list[TokenLengthDistribution] = Field(default_factory=list)
    speed_metrics: list[SpeedMetric] = Field(default_factory=list)


###############################################################################
class GlobalMetrics(BaseModel):
    tokenizer: str
    dataset_name: str
    tokenization_speed_tps: float = Field(default=0.0)
    throughput_chars_per_sec: float = Field(default=0.0)
    processing_time_seconds: float = Field(default=0.0)
    vocabulary_size: int = Field(default=0)
    avg_sequence_length: float = Field(default=0.0)
    median_sequence_length: float = Field(default=0.0)
    subword_fertility: float = Field(default=0.0)
    oov_rate: float = Field(default=0.0)
    word_recovery_rate: float = Field(default=0.0)
    character_coverage: float = Field(default=0.0)
    determinism_rate: float = Field(default=0.0)
    boundary_preservation_rate: float = Field(default=0.0)
    round_trip_fidelity_rate: float = Field(default=0.0)
    model_size_mb: float = Field(default=0.0)
    segmentation_consistency: float = Field(default=0.0)
    token_distribution_entropy: float = Field(default=0.0)
    rare_token_tail_1: int = Field(default=0)
    rare_token_tail_2: int = Field(default=0)
    compression_chars_per_token: float = Field(default=0.0)
    compression_bytes_per_character: float = Field(default=0.0)
    round_trip_text_fidelity_rate: float = Field(default=0.0)
    token_id_ordering_monotonicity: float = Field(default=0.0)
    token_unigram_coverage: float = Field(default=0.0)


###############################################################################
class BenchmarkRunRequest(BaseModel):
    tokenizers: list[str] = Field(
        ..., description="List of tokenizer IDs to benchmark"
    )
    dataset_name: str = Field(..., description="Name of the dataset to use")
    max_documents: int = Field(
        default=0,
        ge=0,
        description="Maximum documents to process (0 = all)",
    )
    custom_tokenizer_name: str | None = Field(
        default=None,
        description="Name of uploaded custom tokenizer to include",
    )
    run_name: str | None = Field(
        default=None,
        description="Optional user-defined benchmark run name",
    )
    selected_metric_keys: list[str] | None = Field(
        default=None,
        description="Optional metric keys selected by the benchmark wizard",
    )


###############################################################################
class BenchmarkPerDocumentTokenizerStats(BaseModel):
    tokenizer: str
    tokens_count: list[int] = Field(default_factory=list)
    tokens_to_words_ratio: list[float] = Field(default_factory=list)
    bytes_per_token: list[float] = Field(default_factory=list)
    boundary_preservation_rate: list[float] = Field(default_factory=list)
    round_trip_token_fidelity: list[float] = Field(default_factory=list)
    round_trip_text_fidelity: list[float] = Field(default_factory=list)
    determinism_stability: list[float] = Field(default_factory=list)
    bytes_per_character: list[float] = Field(default_factory=list)


###############################################################################
class BenchmarkMetricCatalogMetric(BaseModel):
    key: str
    label: str
    description: str
    scope: str
    value_kind: str
    core: bool = Field(default=False)


class BenchmarkMetricCatalogCategory(BaseModel):
    category_key: str
    category_label: str
    metrics: list[BenchmarkMetricCatalogMetric] = Field(default_factory=list)


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


class BenchmarkReportListResponse(BaseModel):
    reports: list[BenchmarkReportSummary] = Field(default_factory=list)


###############################################################################
class BenchmarkRunResponse(BaseModel):
    status: str = Field(default="success")
    report_id: int | None = Field(default=None)
    report_version: int = Field(default=1)
    created_at: str | None = Field(default=None)
    run_name: str | None = Field(default=None)
    selected_metric_keys: list[str] = Field(default_factory=list)
    dataset_name: str = Field(..., description="Name of the benchmarked dataset")
    documents_processed: int = Field(..., description="Number of documents processed")
    tokenizers_processed: list[str] = Field(
        default_factory=list, description="List of successfully processed tokenizers"
    )
    tokenizers_count: int = Field(default=0, description="Number of tokenizers benchmarked")
    global_metrics: list[GlobalMetrics] = Field(
        default_factory=list, description="Global benchmark metrics per tokenizer"
    )
    chart_data: ChartData = Field(
        default_factory=ChartData, description="Structured data for frontend charts"
    )
    per_document_stats: list[BenchmarkPerDocumentTokenizerStats] = Field(
        default_factory=list,
        description="Per-document tokenizer statistics arrays for dispersion views",
    )

