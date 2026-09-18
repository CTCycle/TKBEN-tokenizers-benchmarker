import { describe, expect, it } from 'vitest';
import { dashboardFileName } from './dashboard-file-name';

describe('dashboardFileName', () => {
  it('removes path separators from dataset and tokenizer identifiers', () => {
    expect(dashboardFileName('dataset', 'custom/dataset_smoke')).toBe('dataset-custom-dataset_smoke-report.pdf');
    expect(dashboardFileName('tokenizer', 'owner/model', 'report-7')).toBe('tokenizer-owner-model-report-7.pdf');
  });

  it('falls back to a safe identifier when the input contains no filename characters', () => {
    expect(dashboardFileName('dataset', '///')).toBe('dataset-dashboard-report.pdf');
  });
});
