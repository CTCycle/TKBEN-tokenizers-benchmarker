from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from TKBEN.server.common.utils.security import contains_control_chars


###############################################################################
class HFAccessKeyCreateRequest(BaseModel):
    key_value: str = Field(..., description="Raw Hugging Face access key.")

    # -------------------------------------------------------------------------
    @field_validator("key_value")
    @classmethod
    def validate_key_value(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Hugging Face key must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ValueError("Hugging Face key cannot be empty.")
        if len(normalized) > 512:
            raise ValueError("Hugging Face key is too long.")
        if " " in normalized:
            raise ValueError("Hugging Face key must not contain whitespace.")
        if contains_control_chars(normalized):
            raise ValueError("Hugging Face key contains unsupported characters.")
        if not normalized.startswith("hf_"):
            raise ValueError("Hugging Face key must start with 'hf_'.")
        return normalized


###############################################################################
class HFAccessKeyListItem(BaseModel):
    id: int
    created_at: datetime
    is_active: bool
    masked_preview: str


###############################################################################
class HFAccessKeyListResponse(BaseModel):
    keys: list[HFAccessKeyListItem] = Field(default_factory=list)


###############################################################################
class HFAccessKeyRevealResponse(BaseModel):
    id: int
    key_value: str


###############################################################################
class HFAccessKeyDeleteResponse(BaseModel):
    status: str = Field(default="success")
    message: str = Field(default="Key removed.")


###############################################################################
class HFAccessKeyActivateResponse(BaseModel):
    status: str = Field(default="success")
    message: str = Field(default="Active key updated.")
