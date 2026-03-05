import { API_ENDPOINTS } from '../constants';
import type { DashboardType } from '../types/api';

export interface ExportDashboardRequest {
    dashboardType: DashboardType;
    reportName: string;
    fileName: string;
    dashboardPayload: Record<string, unknown>;
}

export interface ExportDashboardResult {
    fileName: string;
    pageCount: number;
    blob: Blob;
}

export async function exportDashboardPdf(
    request: ExportDashboardRequest,
): Promise<ExportDashboardResult> {
    const response = await fetch(API_ENDPOINTS.EXPORTS_DASHBOARD_PDF, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            dashboard_type: request.dashboardType,
            report_name: request.reportName,
            file_name: request.fileName,
            dashboard_payload: request.dashboardPayload,
        }),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to export dashboard: ${response.status}`);
    }

    const disposition = response.headers.get('content-disposition') ?? '';
    const fileNameMatch = disposition.match(/filename="([^"]+)"/i);
    const fileName = fileNameMatch?.[1]?.trim() || request.fileName;
    const pageCountHeader = response.headers.get('X-Export-Page-Count');
    const pageCount = pageCountHeader ? Number(pageCountHeader) : 1;

    return {
        fileName,
        pageCount: Number.isFinite(pageCount) && pageCount > 0 ? pageCount : 1,
        blob: await response.blob(),
    };
}
