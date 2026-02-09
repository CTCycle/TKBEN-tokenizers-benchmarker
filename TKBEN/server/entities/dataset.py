from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


###############################################################################
class DatasetPayload(BaseModel):
    columns: list[str] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(default_factory=list)


###############################################################################
class HistogramData(BaseModel):
    bins: list[str] = Field(default_factory=list, description="Bin labels for histogram")
    counts: list[int] = Field(default_factory=list, description="Count per bin")
    bin_edges: list[float] = Field(default_factory=list, description="Numeric bin edges")
    min_length: int = Field(default=0, description="Minimum document length")
    max_length: int = Field(default=0, description="Maximum document length")
    mean_length: float = Field(default=0.0, description="Mean document length")
    median_length: float = Field(default=0.0, description="Median document length")


###############################################################################
class DatasetDownloadConfigs(BaseModel):
    configuration: str | None = Field(
        default=None, description="Optional dataset configuration/subset name"
    )


###############################################################################
class DatasetDownloadRequest(BaseModel):
    corpus: str = Field(..., description="HuggingFace dataset corpus identifier")
    configs: DatasetDownloadConfigs = Field(
        ..., description="Dataset download options. Use configs.configuration when needed."
    )
    hf_access_token: str | None = Field(
        default=None, description="Optional HuggingFace access token"
    )


###############################################################################
class DatasetDownloadResponse(BaseModel):
    status: str = Field(default="success")
    dataset_name: str = Field(..., description="Full dataset name (corpus/config)")
    text_column: str = Field(..., description="Column used for text extraction")
    document_count: int = Field(..., description="Total number of documents in dataset")
    saved_count: int = Field(..., description="Number of documents saved to database")
    histogram: HistogramData = Field(..., description="Document length distribution")


###############################################################################
class DatasetLoadResponse(BaseModel):
    status: str = Field(default="success")
    summary: str = Field(default="")
    dataset: DatasetPayload | dict[str, Any] | None = None


###############################################################################
class CustomDatasetUploadResponse(BaseModel):
    """Response schema for custom dataset file uploads."""

    status: str = Field(default="success")
    dataset_name: str = Field(..., description="Name derived from uploaded file")
    text_column: str = Field(..., description="Column used for text extraction")
    document_count: int = Field(..., description="Total documents in dataset")
    saved_count: int = Field(..., description="Documents saved to database")
    histogram: HistogramData = Field(..., description="Document length distribution")


###############################################################################
class DatasetAnalysisRequest(BaseModel):
    """Request schema for dataset analysis."""

    dataset_name: str = Field(..., description="Name of dataset to analyze")


###############################################################################
class WordFrequency(BaseModel):
    word: str = Field(..., description="Word token")
    count: int = Field(..., description="Frequency count")


###############################################################################
class DatasetStatisticsSummary(BaseModel):
    """Summary of word-level statistics from dataset analysis."""

    total_documents: int = Field(default=0, description="Number of analyzed documents")
    mean_words_count: float = Field(default=0.0, description="Mean word count per document")
    median_words_count: float = Field(default=0.0, description="Median word count per document")
    mean_avg_word_length: float = Field(default=0.0, description="Mean average word length")
    mean_std_word_length: float = Field(default=0.0, description="Mean std deviation of word length")


###############################################################################
class DatasetAnalysisResponse(BaseModel):
    """Response schema for dataset analysis endpoint."""

    status: str = Field(default="success")
    dataset_name: str = Field(..., description="Name of analyzed dataset")
    document_count: int = Field(..., description="Number of documents analyzed")
    document_length_histogram: HistogramData = Field(
        ..., description="Document length histogram"
    )
    word_length_histogram: HistogramData = Field(
        ..., description="Word length histogram"
    )
    min_document_length: int = Field(default=0, description="Minimum document length")
    max_document_length: int = Field(default=0, description="Maximum document length")
    most_common_words: list[WordFrequency] = Field(
        default_factory=list, description="Top 10 most common words"
    )
    least_common_words: list[WordFrequency] = Field(
        default_factory=list, description="Top 10 least common words"
    )


###############################################################################
class DatasetPreview(BaseModel):
    dataset_name: str = Field(..., description="Dataset identifier")
    document_count: int = Field(..., description="Number of documents")


###############################################################################
class DatasetListResponse(BaseModel):
    """Response schema for listing available datasets."""

    datasets: list[DatasetPreview] = Field(
        default_factory=list, description="List of dataset names in the database"
    )


