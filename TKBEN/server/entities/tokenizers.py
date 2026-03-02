from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from TKBEN.server.common.utils.security import normalize_identifier


###############################################################################
class TokenizerSignature(BaseModel):
    identifier: str
    records: list[dict[str, Any]] = Field(default_factory=list)

    # -------------------------------------------------------------------------
    @field_validator("identifier")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        return normalize_identifier(value, "Tokenizer identifier", max_length=160)


###############################################################################
class TokenizerScanRequest(BaseModel):
    limit: int = Field(default=100, ge=1, le=1000)


###############################################################################
class TokenizerScanResponse(BaseModel):
    status: str = Field(default="success")
    identifiers: list[str] = Field(default_factory=list)
    count: int = Field(default=0)


###############################################################################
class TokenizerListItem(BaseModel):
    tokenizer_name: str


###############################################################################
class TokenizerListResponse(BaseModel):
    tokenizers: list[TokenizerListItem] = Field(default_factory=list)
    count: int = Field(default=0)


###############################################################################
class TokenizerDownloadRequest(BaseModel):
    tokenizers: list[str] = Field(
        default_factory=list,
        description="Tokenizer IDs to download and persist",
    )

    # -------------------------------------------------------------------------
    @field_validator("tokenizers")
    @classmethod
    def validate_tokenizers(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for tokenizer in value:
            cleaned = normalize_identifier(
                tokenizer,
                "Tokenizer identifier",
                max_length=160,
            )
            if cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        if len(normalized) > 200:
            raise ValueError("Too many tokenizers requested (max 200).")
        return normalized


###############################################################################
class TokenizerDownloadResponse(BaseModel):
    status: str = Field(default="success")
    downloaded: list[str] = Field(default_factory=list)
    already_downloaded: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    requested_count: int = Field(default=0)
    downloaded_count: int = Field(default=0)
    already_downloaded_count: int = Field(default=0)
    failed_count: int = Field(default=0)


###############################################################################
class TokenizerSettingsResponse(BaseModel):
    default_scan_limit: int
    max_scan_limit: int
    min_scan_limit: int


###############################################################################
class TokenizerUploadResponse(BaseModel):
    """Response schema for custom tokenizer upload."""

    status: str = Field(default="success")
    tokenizer_name: str = Field(..., description="Name assigned to uploaded tokenizer")
    is_compatible: bool = Field(..., description="Whether tokenizer is compatible")


###############################################################################
class TokenizerReportGenerateRequest(BaseModel):
    tokenizer_name: str = Field(..., description="Persisted tokenizer name")

    # -------------------------------------------------------------------------
    @field_validator("tokenizer_name")
    @classmethod
    def validate_tokenizer_name(cls, value: str) -> str:
        return normalize_identifier(value, "Tokenizer name", max_length=160)


###############################################################################
class TokenizerLengthHistogram(BaseModel):
    bins: list[str] = Field(default_factory=list)
    counts: list[int] = Field(default_factory=list)
    bin_edges: list[float] = Field(default_factory=list)
    min_length: int = Field(default=0)
    max_length: int = Field(default=0)
    mean_length: float = Field(default=0.0)
    median_length: float = Field(default=0.0)


###############################################################################
class TokenizerReportResponse(BaseModel):
    status: str = Field(default="success")
    report_id: int
    report_version: int = Field(default=1)
    created_at: str
    tokenizer_name: str
    description: str | None = None
    huggingface_url: str | None = None
    global_stats: dict[str, Any] = Field(default_factory=dict)
    token_length_histogram: TokenizerLengthHistogram = Field(
        default_factory=TokenizerLengthHistogram
    )
    vocabulary_size: int = Field(default=0)


###############################################################################
class TokenizerVocabularyItem(BaseModel):
    token_id: int
    token: str
    length: int


###############################################################################
class TokenizerVocabularyPageResponse(BaseModel):
    status: str = Field(default="success")
    report_id: int
    tokenizer_name: str
    offset: int
    limit: int
    total: int
    items: list[TokenizerVocabularyItem] = Field(default_factory=list)

