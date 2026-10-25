// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { describe, expect, it } from 'vitest';
import type { JobStartResponse, JobStatusResponse } from '../api/api.models';
import { JobsStore } from './jobs.store';

const start = (id: string, type = 'benchmark_run'): JobStartResponse => ({ job_id: id, job_type: type, status: 'pending', message: '', poll_interval: 1 });
const status = (id: string, state: string, progress: number, error?: string): JobStatusResponse => ({ job_id: id, job_type: 'benchmark_run', status: state, progress, error });

describe('JobsStore', () => {
  it('tracks progress, completion, and failures per job', () => {
    const store = TestBed.inject(JobsStore);
    store.track(start('a'));
    store.track(start('b', 'dataset_download'));
    expect(store.activeCount()).toBe(2);
    expect(store.jobs()[1]?.label).toBe('Cross benchmark');

    store.update(status('a', 'running', 40));
    expect(store.jobs().find((job) => job.id === 'a')?.progress).toBe(40);

    store.update(status('a', 'completed', 99));
    store.update(status('b', 'failed', 10, 'boom'));
    expect(store.jobs().find((job) => job.id === 'a')).toMatchObject({ state: 'completed', progress: 100 });
    expect(store.jobs().find((job) => job.id === 'b')).toMatchObject({ state: 'failed', error: 'boom' });
    expect(store.activeCount()).toBe(0);

    store.update(status('a', 'running', 10));
    expect(store.jobs().find((job) => job.id === 'a')?.state).toBe('completed');

    store.clearFinished();
    expect(store.jobs()).toEqual([]);
  });

  it('restores persisted jobs without duplicating tracked ones', () => {
    const store = TestBed.inject(JobsStore);
    store.track(start('live'));
    store.restore([
      { job_id: 'live', job_type: 'benchmark_run', status: 'running', progress: 5, created_at: 1 },
      { job_id: 'old', job_type: 'tokenizer_report', status: 'completed', progress: 100, created_at: 2 },
      { job_id: 'bad', job_type: 'dataset_validation', status: 'failed', progress: 30, error: 'boom', created_at: 3 },
    ]);

    expect(store.jobs().map((job) => job.id)).toEqual(['live', 'bad', 'old']);
    expect(store.jobs().find((job) => job.id === 'old')).toMatchObject({ state: 'completed', label: 'Tokenizer report', startedAt: 2000 });
    expect(store.jobs().find((job) => job.id === 'bad')).toMatchObject({ state: 'failed', error: 'boom' });
    expect(store.activeCount()).toBe(1);
  });
});
