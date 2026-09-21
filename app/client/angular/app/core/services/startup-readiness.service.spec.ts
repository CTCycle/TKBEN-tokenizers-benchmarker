import { HttpClient } from '@angular/common/http';
import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { StartupReadinessService } from './startup-readiness.service';

describe('StartupReadinessService', () => {
  afterEach(() => {
    TestBed.resetTestingModule();
    vi.useRealTimers();
  });

  it('reveals the application after the health endpoint reports ok', () => {
    vi.useFakeTimers();
    const http = { get: vi.fn().mockReturnValue(of({ status: 'ok' })) };
    TestBed.configureTestingModule({
      providers: [
        StartupReadinessService,
        { provide: HttpClient, useValue: http },
      ],
    });

    const service = TestBed.inject(StartupReadinessService);
    service.start();
    vi.advanceTimersByTime(0);

    expect(service.status()).toBe('ready');
    expect(http.get).toHaveBeenCalledTimes(1);

    vi.advanceTimersByTime(5_000);
    expect(http.get).toHaveBeenCalledTimes(1);
  });

  it('keeps polling quietly and reports a slow startup before failing', () => {
    vi.useFakeTimers();
    const http = {
      get: vi.fn().mockReturnValue(throwError(() => new Error('backend unavailable'))),
    };
    TestBed.configureTestingModule({
      providers: [
        StartupReadinessService,
        { provide: HttpClient, useValue: http },
      ],
    });

    const service = TestBed.inject(StartupReadinessService);
    service.start();
    vi.advanceTimersByTime(15_000);

    expect(service.status()).toBe('slow');
    expect(service.errorMessage()).toBeNull();

    vi.advanceTimersByTime(45_000);

    expect(service.status()).toBe('failed');
    expect(service.errorMessage()).toContain('expected startup window');
    const attemptsAfterFailure = http.get.mock.calls.length;
    vi.advanceTimersByTime(5_000);
    expect(http.get).toHaveBeenCalledTimes(attemptsAfterFailure);
  });

  it('does not create duplicate polling and retry starts a fresh attempt', () => {
    vi.useFakeTimers();
    const http = {
      get: vi.fn()
        .mockReturnValueOnce(throwError(() => new Error('backend unavailable')))
        .mockReturnValue(of({ status: 'ok' })),
    };
    TestBed.configureTestingModule({
      providers: [
        StartupReadinessService,
        { provide: HttpClient, useValue: http },
      ],
    });

    const service = TestBed.inject(StartupReadinessService);
    service.start();
    service.start();
    vi.advanceTimersByTime(0);
    expect(http.get).toHaveBeenCalledTimes(1);
    expect(service.status()).toBe('checking');

    service.retry();
    vi.advanceTimersByTime(0);

    expect(http.get).toHaveBeenCalledTimes(2);
    expect(service.status()).toBe('ready');
  });

  it('cleans up the poller when the injection context is destroyed', () => {
    vi.useFakeTimers();
    const http = { get: vi.fn().mockReturnValue(throwError(() => new Error('offline'))) };
    TestBed.configureTestingModule({
      providers: [
        StartupReadinessService,
        { provide: HttpClient, useValue: http },
      ],
    });

    const service = TestBed.inject(StartupReadinessService);
    service.start();
    vi.advanceTimersByTime(0);
    const attemptsBeforeDestroy = http.get.mock.calls.length;

    TestBed.resetTestingModule();
    vi.advanceTimersByTime(5_000);

    expect(service.status()).toBe('checking');
    expect(http.get).toHaveBeenCalledTimes(attemptsBeforeDestroy);
  });
});
