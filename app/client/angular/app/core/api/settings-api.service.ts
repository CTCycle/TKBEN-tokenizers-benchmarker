import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import type {
  RuntimeSettingsPatchRequest,
  RuntimeSettingsResetRequest,
  RuntimeSettingsResponse,
} from './api.models';

@Injectable({ providedIn: 'root' })
export class SettingsApiService {
  private readonly http = inject(HttpClient);

  get(): Observable<RuntimeSettingsResponse> {
    return this.http.get<RuntimeSettingsResponse>('/api/settings');
  }

  patch(request: RuntimeSettingsPatchRequest): Observable<RuntimeSettingsResponse> {
    return this.http.patch<RuntimeSettingsResponse>('/api/settings', request);
  }

  reset(request: RuntimeSettingsResetRequest): Observable<RuntimeSettingsResponse> {
    return this.http.post<RuntimeSettingsResponse>('/api/settings/reset', request);
  }
}
