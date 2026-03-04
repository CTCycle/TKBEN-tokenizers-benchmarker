from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DashboardType = Literal["dataset", "tokenizer", "benchmark"]


###############################################################################
class DashboardExportResponse(BaseModel):
    status: str = Field(default="success")
    dashboard_type: DashboardType
    output_path: str
    file_name: str
    page_count: int = Field(default=1, ge=1)
    image_width: int = Field(default=0, ge=1)
    image_height: int = Field(default=0, ge=1)

