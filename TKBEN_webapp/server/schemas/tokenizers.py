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

