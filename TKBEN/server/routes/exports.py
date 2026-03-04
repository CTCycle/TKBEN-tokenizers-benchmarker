from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from TKBEN.server.common.constants import (
    API_ROUTE_EXPORTS_DASHBOARD_PDF,
    API_ROUTER_PREFIX_EXPORTS,
)
from TKBEN.server.entities.exports import DashboardExportResponse
from TKBEN.server.services.dashboard_export import DashboardExportService

router = APIRouter(prefix=API_ROUTER_PREFIX_EXPORTS, tags=["exports"])


###############################################################################
@router.post(
    API_ROUTE_EXPORTS_DASHBOARD_PDF,
    response_model=DashboardExportResponse,
    status_code=status.HTTP_200_OK,
)
async def export_dashboard_pdf(
    dashboard_type: str = Form(...),
    report_name: str = Form(""),
    output_dir: str = Form(...),
    file_name: str = Form(...),
    image_png: UploadFile = File(...),
) -> DashboardExportResponse:
    content_type = (image_png.content_type or "").lower()
    if content_type and content_type not in {"image/png", "application/octet-stream"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dashboard capture must be a PNG image.",
        )

    image_bytes = await image_png.read()
    service = DashboardExportService()
    try:
        result = await asyncio.to_thread(
            service.export_dashboard_pdf,
            dashboard_type=dashboard_type,
            report_name=report_name,
            output_dir=output_dir,
            file_name=file_name,
            image_bytes=image_bytes,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export dashboard PDF.",
        ) from exc

    return DashboardExportResponse(
        status="success",
        dashboard_type=result.dashboard_type,
        output_path=result.output_path,
        file_name=result.file_name,
        page_count=result.page_count,
        image_width=result.image_width,
        image_height=result.image_height,
    )
