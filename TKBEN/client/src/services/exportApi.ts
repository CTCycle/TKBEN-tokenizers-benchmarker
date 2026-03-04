import { API_ENDPOINTS } from '../constants';
import type { DashboardExportResponse, DashboardType } from '../types/api';

export interface ExportDashboardRequest {
    dashboardType: DashboardType;
    reportName: string;
    outputDir: string;
    fileName: string;
    imagePng: Blob;
}

export async function exportDashboardPdf(
    request: ExportDashboardRequest,
): Promise<DashboardExportResponse> {
    const formData = new FormData();
    formData.append('dashboard_type', request.dashboardType);
    formData.append('report_name', request.reportName);
    formData.append('output_dir', request.outputDir);
    formData.append('file_name', request.fileName);
    formData.append('image_png', request.imagePng, `${request.dashboardType}-dashboard.png`);

    const response = await fetch(API_ENDPOINTS.EXPORTS_DASHBOARD_PDF, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Failed to export dashboard: ${response.status}`);
    }

    return response.json();
}

