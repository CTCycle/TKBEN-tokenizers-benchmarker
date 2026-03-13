from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Response, status

from TKBEN.server.common.constants import (
    API_ROUTE_EXPORTS_DASHBOARD_PDF,
    API_ROUTER_PREFIX_EXPORTS,
)
from TKBEN.server.entities.exports import DashboardExportRequest
from TKBEN.server.services.export import DashboardExportService

router = APIRouter(prefix=API_ROUTER_PREFIX_EXPORTS, tags=["exports"])


###############################################################################
@router.post(
    API_ROUTE_EXPORTS_DASHBOARD_PDF,
    status_code=status.HTTP_200_OK,
)
async def export_dashboard_pdf(
    request: DashboardExportRequest,
) -> Response:
    service = DashboardExportService()
    try:
        result = await asyncio.to_thread(
            service.export_dashboard_pdf,
            dashboard_type=request.dashboard_type,
            report_name=request.report_name,
            file_name=request.file_name,
            dashboard_payload=request.dashboard_payload,
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

    headers = {
        "Content-Disposition": f'attachment; filename="{result.file_name}"',
        "X-Export-Page-Count": str(result.page_count),
    }
    return Response(
        content=result.pdf_bytes,
        media_type="application/pdf",
        headers=headers,
    )
