import { HttpClient } from '@angular/common/http';
import { DestroyRef, Injectable, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subscription, of, timer } from 'rxjs';
import { catchError, exhaustMap, map, timeout } from 'rxjs/operators';
import type { HealthResponse } from '../api/api.models';

export type StartupStatus = 'checking' | 'slow' | 'failed' | 'ready';

const POLL_INTERVAL_MS = 1_000;
const HEALTH_REQUEST_TIMEOUT_MS = 2_000;
const SLOW_STARTUP_AFTER_MS = 15_000;
const STARTUP_FAILURE_AFTER_MS = 60_000;

@Injectable({ providedIn: 'root' })
export class StartupReadinessService {
  private readonly http = inject(HttpClient);
  private readonly destroyRef = inject(DestroyRef);
  private pollingSubscription: Subscription | null = null;
  private startedAt = 0;

  readonly status = signal<StartupStatus>('checking');
  readonly errorMessage = signal<string | null>(null);

  start(): void {
    if (this.pollingSubscription !== null || this.status() === 'ready') return;

    this.status.set('checking');
    this.errorMessage.set(null);
    this.startedAt = Date.now();
    this.pollingSubscription = timer(0, POLL_INTERVAL_MS).pipe(
      exhaustMap(() => this.http.get<HealthResponse>('/api/health').pipe(
        timeout({ each: HEALTH_REQUEST_TIMEOUT_MS }),
        map((response) => response.status === 'ok'),
        catchError(() => of(false)),
      )),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe((healthy) => {
      if (healthy) {
        this.status.set('ready');
        this.stopPolling();
        return;
      }

      const elapsed = Date.now() - this.startedAt;
      if (elapsed >= STARTUP_FAILURE_AFTER_MS) {
        this.status.set('failed');
        this.errorMessage.set(
          'The backend did not become ready within the expected startup window. Check the launcher output or backend logs, then retry the connection.',
        );
        this.stopPolling();
      } else if (elapsed >= SLOW_STARTUP_AFTER_MS) {
        this.status.set('slow');
      }
    });
  }

  retry(): void {
    if (this.status() === 'ready') return;
    this.stopPolling();
    this.start();
  }

  private stopPolling(): void {
    this.pollingSubscription?.unsubscribe();
    this.pollingSubscription = null;
  }
}
