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
});
