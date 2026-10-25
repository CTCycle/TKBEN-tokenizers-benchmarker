// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, DestroyRef, HostListener, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DecimalPipe } from '@angular/common';
import { JobsApiService } from '../core/api/jobs-api.service';
import { JobsStore, type TrackedJob } from '../core/state/jobs.store';

@Component({
  selector: 'app-jobs-panel',
  imports: [DecimalPipe],
  templateUrl: './jobs-panel.component.html',
})
export class JobsPanelComponent {
  protected readonly jobsStore = inject(JobsStore);
  private readonly jobsApi = inject(JobsApiService);
  private readonly destroyRef = inject(DestroyRef);
  protected readonly open = signal(false);

  constructor() {
    this.jobsApi.restore().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: () => this.jobsStore.jobs()
        .filter((job) => job.state === 'running')
        .forEach((job) => this.jobsApi.watch(job.id).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({ error: () => undefined })),
      error: () => undefined,
    });
  }

  protected toggle(): void {
    this.open.update((value) => !value);
  }

  @HostListener('document:keydown.escape')
  protected close(): void {
    this.open.set(false);
  }

  protected cancel(job: TrackedJob): void {
    if (job.cancelRequested) return;
    this.jobsApi.cancel(job.id).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({ error: () => undefined });
  }

  protected stateLabel(job: TrackedJob): string {
    if (job.state === 'running') return job.cancelRequested ? 'Cancelling…' : 'Running';
    return job.state === 'completed' ? 'Completed' : job.state === 'cancelled' ? 'Cancelled' : 'Failed';
  }
}
