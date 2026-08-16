import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, map } from 'rxjs';
import type { DashboardType } from './api.models';

export interface ExportDashboardRequest {
  readonly dashboardType: DashboardType;
  readonly reportName: string;
  readonly fileName: string;
  readonly dashboardPayload: Record<string, unknown>;
}

export interface ExportDashboardResult {
  readonly fileName: string;
  readonly pageCount: number;
  readonly blob: Blob;
}

@Injectable({ providedIn: 'root' })
export class ExportApiService {
  private readonly http = inject(HttpClient);

  dashboardPdf(request: ExportDashboardRequest): Observable<ExportDashboardResult> {
    return this.http.post('/api/exports/dashboard/pdf', {
      dashboard_type: request.dashboardType,
      report_name: request.reportName,
      file_name: request.fileName,
      dashboard_payload: request.dashboardPayload,
    }, { observe: 'response', responseType: 'blob' }).pipe(
      map((response) => {
        const disposition = response.headers.get('content-disposition') ?? '';
        const fileName = disposition.match(/filename="([^"]+)"/i)?.[1]?.trim() || request.fileName;
        const pageCount = Number(response.headers.get('X-Export-Page-Count') || 1);
        return { fileName, pageCount: Number.isFinite(pageCount) && pageCount > 0 ? pageCount : 1, blob: response.body ?? new Blob() };
      }),
    );
  }
}
