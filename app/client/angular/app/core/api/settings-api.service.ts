// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import type {
  DatabaseConfiguration,
  DatabaseConfigurationRequest,
  DatabaseConnectionTestResponse,
  PathSettings,
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

  getPaths(): Observable<PathSettings> {
    return this.http.get<PathSettings>('/api/settings/paths');
  }

  putPaths(request: Pick<PathSettings, 'data_dir' | 'log_dir'>): Observable<PathSettings> {
    return this.http.put<PathSettings>('/api/settings/paths', request);
  }

  getDatabase():Observable<DatabaseConfiguration> {
    return this.http.get<DatabaseConfiguration>('/api/database');
  }

  testDatabase(request: DatabaseConfigurationRequest): Observable<DatabaseConnectionTestResponse> {
    return this.http.post<DatabaseConnectionTestResponse>('/api/database/test', request);
  }

  switchDatabase(request: DatabaseConfigurationRequest): Observable<DatabaseConfiguration> {
    return this.http.post<DatabaseConfiguration>('/api/database/switch', request);
  }
}
