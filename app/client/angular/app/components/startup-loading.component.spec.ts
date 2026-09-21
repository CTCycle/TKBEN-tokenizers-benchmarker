import { TestBed } from '@angular/core/testing';
import { afterEach, describe, expect, it } from 'vitest';
import { StartupLoadingComponent } from './startup-loading.component';

describe('StartupLoadingComponent', () => {
  afterEach(() => TestBed.resetTestingModule());

  it('renders the technical loading flow and exposes a retry action on failure', () => {
    const fixture = TestBed.createComponent(StartupLoadingComponent);
    let retryCount = 0;
    fixture.componentInstance.retry.subscribe(() => retryCount++);
    fixture.componentRef.setInput('status', 'failed');
    fixture.componentRef.setInput('errorMessage', 'Backend startup failed.');
    fixture.detectChanges();

    const element = fixture.nativeElement as HTMLElement;
    expect(element.querySelector('h1')?.textContent).toContain('Preparing tokenizer benchmarks');
    expect(element.querySelector('.startup-token-stage')).not.toBeNull();
    expect(element.querySelector('.startup-chart')).not.toBeNull();
    expect(element.querySelector('[role="alert"]')?.textContent).toContain('Backend startup failed.');

    element.querySelector<HTMLButtonElement>('button')?.click();
    expect(retryCount).toBe(1);
  });
});
