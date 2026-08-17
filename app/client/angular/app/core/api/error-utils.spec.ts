import { describe, expect, it } from 'vitest';
import { errorMessage, errorMessageAsync, formatApiError } from './error-utils';

describe('API error normalization', () => {
  it('formats FastAPI array details with field locations', () => {
    expect(formatApiError([{ loc: ['body', 'dataset_name'], msg: 'Field required' }], 'fallback')).toBe('body.dataset_name: Field required');
  });

  it('keeps a useful HttpClient-style error detail', () => {
    expect(errorMessage({ error: { detail: 'Dataset not found' } }, 'fallback')).toBe('Dataset not found');
    expect(errorMessage(new Error('network failed'), 'fallback')).toBe('network failed');
  });

  it('decodes structured Blob errors used by PDF exports', async () => {
    await expect(
      errorMessageAsync(
        { error: new Blob([JSON.stringify({ detail: 'Export unavailable' })], { type: 'application/json' }) },
        'fallback',
      ),
    ).resolves.toBe('Export unavailable');
  });
});
