from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


###############################################################################
class HFAccessKeyCreateRequest(BaseModel):
    key_value: str = Field(..., description="Raw Hugging Face access key.")


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
