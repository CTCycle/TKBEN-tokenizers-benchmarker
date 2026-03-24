from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


DashboardType = Literal["dataset", "tokenizer", "benchmark"]


###############################################################################
class DashboardExportRequest(BaseModel):
    dashboard_type: DashboardType
    report_name: str = Field(default="")
    file_name: str = Field(default="")
    dashboard_payload: dict[str, Any] = Field(default_factory=dict)
