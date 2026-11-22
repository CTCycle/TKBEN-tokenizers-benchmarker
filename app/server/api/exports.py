# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Response, status

from server.common.constants import (
    API_ROUTE_EXPORTS_DASHBOARD_PDF,
    API_ROUTER_PREFIX_EXPORTS,
)
from server.contracts.exports import DashboardExportRequest

if TYPE_CHECKING:
    from server.services.export import DashboardPdfDocument

router = APIRouter(prefix=API_ROUTER_PREFIX_EXPORTS, tags=["exports"])


###############################################################################
def _render_dashboard_pdf(request: DashboardExportRequest) -> DashboardPdfDocument:
    from server.services.export import DashboardExportService

    return DashboardExportService().export_dashboard_pdf(
        dashboard_type=request.dashboard_type,
        report_name=request.report_name,
        file_name=request.file_name,
        dashboard_payload=request.dashboard_payload,
    )


###############################################################################
@router.post(
    API_ROUTE_EXPORTS_DASHBOARD_PDF,
    status_code=status.HTTP_200_OK,
)
async def export_dashboard_pdf(
    request: DashboardExportRequest,
) -> Response:
    try:
        result = await asyncio.to_thread(_render_dashboard_pdf, request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export dashboard as PDF.",
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
