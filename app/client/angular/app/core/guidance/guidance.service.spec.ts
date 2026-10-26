// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { TestBed } from '@angular/core/testing';
import { Router } from '@angular/router';
import { describe, expect, it, vi } from 'vitest';
import { GuidanceService } from './guidance.service';

describe('GuidanceService', () => {
  it('navigates to the tour page before activating it and ignores unknown tours', async () => {
    const navigateByUrl = vi.fn().mockResolvedValue(true);
    TestBed.configureTestingModule({ providers: [{ provide: Router, useValue: { url: '/dataset', navigateByUrl } }] });
    const service = TestBed.inject(GuidanceService);

    await service.start('datasets');
    expect(navigateByUrl).not.toHaveBeenCalled();
    expect(service.activeTour()?.id).toBe('datasets');

    await service.start('tokenizers');
    expect(navigateByUrl).toHaveBeenCalledWith('/tokenizers');
    expect(service.activeTour()?.id).toBe('tokenizers');

    await service.start('missing');
    expect(service.activeTour()?.id).toBe('tokenizers');

    service.stop();
    expect(service.activeTour()).toBeNull();
  });
});
