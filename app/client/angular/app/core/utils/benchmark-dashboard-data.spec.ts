import { describe, expect, it } from 'vitest';
import type { BenchmarkDashboardWidgetData } from '../api/api.models';
import {
  benchmarkColorFor,
  benchmarkComparisonRows,
  benchmarkComparisonTokenizers,
  benchmarkDistributionMedian,
  benchmarkPointValue,
  classifyBenchmarkDataShape,
  distributionViews,
  formatBenchmarkAxisValue,
  formatBenchmarkTooltipValue,
  formatBenchmarkValue,
  pointViews,
  relativeBenchmarkDelta,
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

  it('extracts only finite point and distribution baseline candidates', () => {
    const widget = {
      ...pointWidget,
      points: [
        { tokenizer: 'alpha', value: 10, interval_low: null, interval_high: null },
        { tokenizer: 'invalid-point', value: Number.NaN, interval_low: null, interval_high: null },
      ],
      distributions: [{ tokenizer: 'beta', min: 1, q1: 2, median: 3, q3: 4, max: 5, sample_count: 5 }],
      buckets: [{ tokenizer: 'bucket-only', bucket: 'short', value: 1 }],
      histogram_bins: [{ tokenizer: 'histogram-only', bin_low: 0, bin_high: 1, count: 1, proportion: 1 }],
    };

    expect(benchmarkComparisonTokenizers([widget])).toEqual(['alpha', 'beta']);
    expect(benchmarkPointValue(widget, 'alpha')).toBe(10);
    expect(benchmarkPointValue(widget, 'invalid-point')).toBeNull();
    expect(benchmarkDistributionMedian(widget, 'beta')).toBe(3);
  });

  it('calculates and formats direction-neutral relative deltas', () => {
    expect(relativeBenchmarkDelta(120, 100)).toBe(20);
    expect(relativeBenchmarkDelta(80, 100)).toBe(-20);
    expect(relativeBenchmarkDelta(100, 100)).toBe(0);
    expect(relativeBenchmarkDelta(10, 0)).toBeNull();
    expect(relativeBenchmarkDelta(Number.NaN, 100)).toBeNull();

    const rows = benchmarkComparisonRows({
      ...pointWidget,
      points: [
        { tokenizer: 'alpha', value: 100, interval_low: null, interval_high: null },
        { tokenizer: 'beta', value: 80, interval_low: null, interval_high: null },
      ],
    }, 'alpha');
    expect(rows).toEqual([
      { tokenizer: 'alpha', deltaPercent: null, formattedDelta: 'Baseline', isBaseline: true },
      { tokenizer: 'beta', deltaPercent: -20, formattedDelta: '-20.00%', isBaseline: false },
    ]);
  });

  it('compares distribution medians and leaves bucket and histogram widgets unchanged', () => {
    const distributionWidget = {
      ...pointWidget,
      points: [],
      distributions: [
        { tokenizer: 'alpha', min: 1, q1: 2, median: 3, q3: 4, max: 5, sample_count: 5 },
        { tokenizer: 'beta', min: 2, q1: 3, median: 4, q3: 5, max: 6, sample_count: 5 },
      ],
    };
    expect(benchmarkComparisonRows(distributionWidget, 'alpha')[1]?.formattedDelta).toBe('+33.33%');
    expect(benchmarkComparisonRows({ ...pointWidget, points: [], distributions: [], buckets: [{ tokenizer: 'beta', bucket: 'short', value: 2 }] }, 'alpha')).toEqual([]);
    expect(benchmarkComparisonRows({ ...pointWidget, points: [], distributions: [], histogram_bins: [{ tokenizer: 'beta', bin_low: 0, bin_high: 1, count: 1, proportion: 1 }] }, 'alpha')).toEqual([]);
  });
});
