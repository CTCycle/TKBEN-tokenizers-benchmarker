import { TestBed } from '@angular/core/testing';
import { afterEach, describe, expect, it } from 'vitest';
import { StartupLoadingComponent } from './startup-loading.component';

describe('StartupLoadingComponent', () => {
  afterEach(() => TestBed.resetTestingModule());

  it('renders the compact tokenizer flow and exposes a retry action on failure', () => {
    const fixture = TestBed.createComponent(StartupLoadingComponent);
    let retryCount = 0;
    fixture.componentInstance.retry.subscribe(() => retryCount++);
    fixture.componentRef.setInput('status', 'failed');
    fixture.componentRef.setInput('errorMessage', 'Backend startup failed.');
    fixture.detectChanges();

    const element = fixture.nativeElement as HTMLElement;
    expect(element.querySelector('h1')?.textContent).toBe('TKBEN');
    expect(element.querySelector('#startup-description')?.textContent).toContain(
      'We are loading the service, please wait...',
    );
    expect(element.querySelector('.startup-tokenizer-machine')).not.toBeNull();
    expect(element.querySelector('.startup-lane')).not.toBeNull();
    expect(element.querySelectorAll('.startup-word')).toHaveLength(6);
    expect(element.querySelectorAll('.startup-token')).toHaveLength(12);
    expect(element.textContent).toContain('##ization');
    expect(element.querySelector('.startup-chart')).toBeNull();
    expect(element.querySelector('[role="alert"]')?.textContent).toContain('Backend startup failed.');
    expect(element.querySelector('.startup-screen--failed')).not.toBeNull();

    element.querySelector<HTMLButtonElement>('button')?.click();
    expect(retryCount).toBe(1);
  });
});
