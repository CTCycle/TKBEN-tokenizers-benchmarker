import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { TestBed } from '@angular/core/testing';
import { firstValueFrom, throwError } from 'rxjs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DatasetsApiService } from './datasets-api.service';
import { TokenizersApiService } from './tokenizers-api.service';
import { JobsApiService } from './jobs-api.service';

describe('report API clients', () => {
  afterEach(() => TestBed.resetTestingModule());

  it('normalizes a missing dataset report and preserves encoded names', async () => {
    const http = {
      get: vi.fn().mockReturnValue(throwError(() => new HttpErrorResponse({ status: 404 }))),
    };
    TestBed.configureTestingModule({ providers: [
      DatasetsApiService,
      { provide: JobsApiService, useValue: {} },
      { provide: HttpClient, useValue: http },
    ] });

    const service = TestBed.inject(DatasetsApiService);
    await expect(firstValueFrom(service.latestReport('custom/foo'))).resolves.toBeNull();
    expect(http.get).toHaveBeenCalledWith('/api/datasets/reports/latest?dataset_name=custom%2Ffoo');
  });

  it('normalizes a missing tokenizer report and preserves encoded names', async () => {
    const http = {
      get: vi.fn().mockReturnValue(throwError(() => new HttpErrorResponse({ status: 404 }))),
    };
    TestBed.configureTestingModule({ providers: [
      TokenizersApiService,
      { provide: JobsApiService, useValue: {} },
      { provide: HttpClient, useValue: http },
    ] });

    const service = TestBed.inject(TokenizersApiService);
    await expect(firstValueFrom(service.latestReport('google-bert/bert-base-uncased'))).resolves.toBeNull();
    expect(http.get).toHaveBeenCalledWith('/api/tokenizers/reports/latest?tokenizer_name=google-bert%2Fbert-base-uncased');
  });
});
