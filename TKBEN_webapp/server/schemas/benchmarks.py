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


###############################################################################
class BenchmarkRunResponse(BaseModel):
    status: str = Field(default="success")
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

