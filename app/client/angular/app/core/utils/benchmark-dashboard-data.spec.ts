import { describe, expect, it } from 'vitest';
import type { BenchmarkDashboardWidgetData } from '../api/api.models';
import {
  benchmarkColorFor,
  classifyBenchmarkDataShape,
  distributionViews,
  formatBenchmarkAxisValue,
  formatBenchmarkTooltipValue,
  formatBenchmarkValue,
  pointViews,
  uniqueTokenizers,
} from './benchmark-dashboard-data';

const pointWidget: BenchmarkDashboardWidgetData = {
  widget_id: 'efficiency.speed', metric_keys: ['efficiency.speed'], category_key: 'efficiency', category_label: 'Efficiency', label: 'Speed', description: 'Mean speed', unit: 'tokens/s', display_format: 'number', default_visualization: 'bar', compatible_visualizations: ['bar', 'horizontal_bar'], default_visible: true, width: 'standard',
  points: [{ tokenizer: 'alpha', value: 10, interval_low: 8, interval_high: 12 }], distributions: [], buckets: [], histogram_bins: [],
};

describe('benchmark dashboard chart data', () => {
  it('classifies payload shapes and preserves intervals', () => {
    expect(classifyBenchmarkDataShape(pointWidget)).toBe('point');
    expect(pointViews(pointWidget)[0]?.low).toBe(8);
    expect(pointViews(pointWidget)[0]?.high).toBe(12);
  });

  it('formats units and returns stable colors', () => {
    expect(formatBenchmarkValue(0.25, 'percent')).toBe('25.00%');
    expect(formatBenchmarkValue(12.3456, 'milliseconds')).toContain('ms');
    expect(formatBenchmarkAxisValue(60_000, 'number')).toBe('60k');
    expect(formatBenchmarkAxisValue(0.25, 'milliseconds')).toMatch(/0[,.]25/);
    expect(formatBenchmarkTooltipValue(0.25, 'percent', '%')).toBe('25.00%');
    expect(formatBenchmarkTooltipValue(12.3456, 'number', 'tokens')).toContain('tokens');
    expect(benchmarkColorFor('alpha')).toBe(benchmarkColorFor('alpha'));
  });

  it('uses value fallbacks and deduplicates tokenizers across payload shapes', () => {
    const widget = {
      ...pointWidget,
      points: [{ tokenizer: 'alpha', value: 10, interval_low: null, interval_high: null }],
      distributions: [{ tokenizer: 'alpha', min: 1, q1: 2, median: 3, q3: 4, max: 5, sample_count: 5 }],
      buckets: [{ tokenizer: 'beta', bucket: 'short', value: 1 }],
      histogram_bins: [{ tokenizer: 'beta', bin_low: 0, bin_high: 1, count: 1, proportion: 1 }],
    };

    expect(distributionViews(widget)[0]?.label).toBe('alpha');
    expect(pointViews(widget)[0]?.low).toBe(10);
    expect(uniqueTokenizers(widget)).toEqual(['alpha', 'beta']);
    expect(formatBenchmarkValue(Number.NaN, 'number')).toBe('N/A');
  });
});
