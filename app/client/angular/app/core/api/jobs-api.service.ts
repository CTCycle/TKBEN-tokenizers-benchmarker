// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, catchError, concatMap, filter, map, switchMap, takeWhile, tap, timer, throwError, timeout } from 'rxjs';
import type { JobStartResponse, JobStatusResponse } from './api.models';
import { errorMessage } from './error-utils';
import { JobsStore } from '../state/jobs.store';

export interface JobPollOptions {
  readonly timeoutMs?: number;
}

@Injectable({ providedIn: 'root' })
export class JobsApiService {
  private readonly http = inject(HttpClient);
  private readonly tracker = inject(JobsStore);

  status(jobId: string): Observable<JobStatusResponse> {
    return this.http.get<JobStatusResponse>(`/api/jobs/${encodeURIComponent(jobId)}`);
  }

  cancel(jobId: string): Observable<JobStatusResponse> {
    this.tracker.markCancelRequested(jobId, true);
    return this.http.post<JobStatusResponse>(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {}).pipe(
      tap({ error: () => this.tracker.markCancelRequested(jobId, false) }),
    );
  }

  poll<T>(job: JobStartResponse, options: JobPollOptions = {}): Observable<T> {
    const interval = Math.max(250, Math.round(job.poll_interval * 1000));
    this.tracker.track(job);
    return timer(0, interval).pipe(
      switchMap(() => this.status(job.job_id)),
      map((status) => {
        this.tracker.update(status);
        if (status.status === 'failed') throw new Error(status.error || 'Job failed.');
        if (status.status === 'cancelled') throw new Error('Job was cancelled.');
        return status;
      }),
      takeWhile((status) => status.status !== 'completed', true),
      filter((status) => status.status === 'completed'),
      map((status) => {
        if (status.result === undefined || status.result === null) throw new Error('Job completed without a result payload.');
        return status.result as T;
      }),
      timeout({ first: options.timeoutMs ?? 30 * 60 * 1000 }),
      catchError((error: unknown) => {
        const message = errorMessage(error, 'Job polling failed.');
        this.tracker.fail(job.job_id, message);
        return throwError(() => new Error(message));
      }),
    );
  }

  startAndPoll<T>(start$: Observable<JobStartResponse>, options: JobPollOptions = {}): Observable<T> {
    return start$.pipe(concatMap((job) => this.poll<T>(job, options)));
  }
}
