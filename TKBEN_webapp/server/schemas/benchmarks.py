from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from TKBEN_webapp.server.schemas.dataset import DatasetPayload


###############################################################################
class BenchmarkResults(BaseModel):
    status: str = Field(default="success")
    summary: str
    dataset: DatasetPayload | dict[str, Any] | None = None
