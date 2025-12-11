from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


###############################################################################
class PlotData(BaseModel):
    name: str = Field(..., description="Name of the plot")
    data: str = Field(..., description="Base64-encoded PNG image data")


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
    include_custom_tokenizer: bool = Field(
        default=False,
        description="Include custom tokenizer in benchmarks",
    )
    include_nsl: bool = Field(
        default=False,
        description="Calculate Normalized Sequence Length",
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
    plots: list[PlotData] = Field(
        default_factory=list, description="Generated visualization plots as base64"
    )
    global_metrics: list[GlobalMetrics] = Field(
        default_factory=list, description="Global benchmark metrics per tokenizer"
    )
