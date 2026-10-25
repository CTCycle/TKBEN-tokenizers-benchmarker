// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Injectable, computed, signal } from '@angular/core';
import type { JobStartResponse, JobStatusResponse, JobSummaryResponse } from '../api/api.models';

export type TrackedJobState = 'running' | 'completed' | 'failed' | 'cancelled';

export interface TrackedJob {
  readonly id: string;
  readonly type: string;
  readonly label: string;
  readonly state: TrackedJobState;
  readonly progress: number;
  readonly error: string | null;
  readonly cancelRequested: boolean;
  readonly startedAt: number;
}

const JOB_LABELS: Readonly<Record<string, string>> = {
  dataset_download: 'Dataset download',
  dataset_upload: 'Dataset upload',
  dataset_validation: 'Dataset benchmark',
  tokenizer_download: 'Tokenizer download',
  tokenizer_report: 'Tokenizer report',
  benchmark_run: 'Cross benchmark',
};

function jobState(status: string): TrackedJobState {
  return status === 'completed' || status === 'failed' || status === 'cancelled' ? status : 'running';
}

@Injectable({ providedIn: 'root' })
export class JobsStore {
  readonly jobs = signal<readonly TrackedJob[]>([]);
  readonly activeCount = computed(() => this.jobs().filter((job) => job.state === 'running').length);
  readonly hasFinished = computed(() => this.jobs().some((job) => job.state !== 'running'));

  track(job: JobStartResponse): void {
    const entry: TrackedJob = {
      id: job.job_id,
      type: job.job_type,
      label: JOB_LABELS[job.job_type] ?? job.job_type,
      state: 'running',
      progress: 0,
      error: null,
      cancelRequested: false,
      startedAt: Date.now(),
    };
    this.jobs.update((jobs) => [entry, ...jobs.filter((item) => item.id !== entry.id)]);
  }

  update(status: JobStatusResponse): void {
    this.patch(status.job_id, {
      state: jobState(status.status),
      progress: status.status === 'completed' ? 100 : status.progress,
      error: status.error ?? null,
    });
  }

  restore(summaries: readonly JobSummaryResponse[]): void {
    this.jobs.update((current) => {
      const known = new Set(current.map((job) => job.id));
      const restored = summaries
        .filter((summary) => !known.has(summary.job_id))
        .map((summary): TrackedJob => ({
          id: summary.job_id,
          type: summary.job_type,
          label: JOB_LABELS[summary.job_type] ?? summary.job_type,
          state: jobState(summary.status),
          progress: summary.status === 'completed' ? 100 : summary.progress,
          error: summary.error ?? null,
          cancelRequested: false,
          startedAt: summary.created_at * 1000,
        }));
      return [...current, ...restored].sort((a, b) => b.startedAt - a.startedAt);
    });
  }

  fail(jobId: string, message: string): void {
    this.patch(jobId, { state: 'failed', error: message });
  }

  markCancelRequested(jobId: string, requested: boolean): void {
    this.patch(jobId, { cancelRequested: requested });
  }

  clearFinished(): void {
    this.jobs.update((jobs) => jobs.filter((job) => job.state === 'running'));
  }

  private patch(jobId: string, changes: Partial<TrackedJob>): void {
    this.jobs.update((jobs) => jobs.map((job) => (job.id === jobId && job.state === 'running' ? { ...job, ...changes } : job)));
  }
}
