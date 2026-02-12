from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


###############################################################################
class TokenizerSignature(BaseModel):
    identifier: str
    records: list[dict[str, Any]] = Field(default_factory=list)


###############################################################################
class TokenizerScanRequest(BaseModel):
    limit: int = Field(default=100, ge=1, le=1000)
    hf_access_token: str | None = Field(default=None)


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
    hf_access_token: str | None = Field(default=None)


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

