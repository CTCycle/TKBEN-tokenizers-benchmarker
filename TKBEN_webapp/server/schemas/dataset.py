from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


###############################################################################
class DatasetPayload(BaseModel):
    columns: list[str] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(default_factory=list)


