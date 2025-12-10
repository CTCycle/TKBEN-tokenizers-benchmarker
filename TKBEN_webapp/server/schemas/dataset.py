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
class DatasetDownloadRequest(BaseModel):
    corpus: str = Field(..., description="HuggingFace dataset corpus identifier")
    config: str | None = Field(
        default=None, description="Dataset configuration/subset"
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
