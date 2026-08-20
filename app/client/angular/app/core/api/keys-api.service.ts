import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import type { HFAccessKeyListItem, HFAccessKeyListResponse, HFAccessKeyRevealResponse } from './api.models';

@Injectable({ providedIn: 'root' })
export class KeysApiService {
  private readonly http = inject(HttpClient);

  list(): Observable<HFAccessKeyListResponse> { return this.http.get<HFAccessKeyListResponse>('/api/keys'); }
  add(keyValue: string): Observable<HFAccessKeyListItem> { return this.http.post<HFAccessKeyListItem>('/api/keys', { key_value: keyValue }); }
  activate(id: number): Observable<void> { return this.http.post<void>(`/api/keys/${id}/activate`, {}); }
  deactivate(id: number): Observable<void> { return this.http.post<void>(`/api/keys/${id}/deactivate`, {}); }
  reveal(id: number): Observable<HFAccessKeyRevealResponse> { return this.http.post<HFAccessKeyRevealResponse>(`/api/keys/${id}/reveal`, {}); }
  remove(id: number): Observable<void> { return this.http.delete<void>(`/api/keys/${id}`, { params: { confirm: true } }); }
}
