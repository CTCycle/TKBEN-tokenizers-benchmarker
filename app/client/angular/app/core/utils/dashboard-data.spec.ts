// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { describe, expect, it } from 'vitest';
import type { BenchmarkDashboardWidgetData } from '../api/api.models';
import { benchmarkComparisonRows, benchmarkComparisonTokenizers, benchmarkDistributionMedian, benchmarkPointValue, classifyBenchmarkDataShape, distributionViews, formatBenchmarkAxisValue, formatBenchmarkTooltipValue, formatBenchmarkValue, pointViews, relativeBenchmarkDelta, uniqueTokenizers } from './benchmark-dashboard-data';
import { buildWordCloudFromWordFrequencies, buildZipfCurveFromWordFrequencies, metricDisplayValue, parseWordCloudTerms, parseWordFrequencyItems, parseZipfCurve, toHistogramSeries } from './dataset-dashboard-data';
import { dashboardFileName } from './dashboard-file-name';


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

    expect(benchmarkComparisonTokenizers({
      widgets: [widget],
      available_widget_ids: [],
      available_metric_keys: [],
      unavailable_selected_metric_keys: [],
    })).toEqual(['alpha', 'beta']);
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

describe('dataset dashboard normalization', () => {
  it('rejects malformed frequency rows and keeps positive counts', () => {
    expect(parseWordFrequencyItems([{ word: 'alpha', count: 3 }, { word: '', count: 4 }, { word: 'zero', count: 0 }, null])).toEqual([{ word: 'alpha', count: 3 }]);
  });

  it('normalizes missing cloud weights from counts', () => {
    expect(parseWordCloudTerms([{ word: 'alpha', count: 10 }, { word: 'beta', count: 5 }]).map((item) => item.weight)).toEqual([100, 50]);
  });

  it('sorts and caps deterministic Zipf fallback data', () => {
    expect(buildZipfCurveFromWordFrequencies([{ word: 'b', count: 2 }, { word: 'a', count: 4 }])).toEqual([{ rank: 1, frequency: 4 }, { rank: 2, frequency: 2 }]);
    expect(parseZipfCurve([{ rank: 2, frequency: 4 }, { rank: 1, frequency: 5 }, { rank: 0, frequency: 2 }])).toEqual([{ rank: 1, frequency: 5 }, { rank: 2, frequency: 4 }]);
  });

  it('builds a bounded word-cloud fallback', () => {
    const terms = buildWordCloudFromWordFrequencies(Array.from({ length: 150 }, (_, index) => ({ word: `word-${index}`, count: 150 - index })));
    expect(terms).toHaveLength(120);
    expect(terms[0]?.weight).toBe(100);
  });

  it('keeps dashboard placeholders and histogram bins stable with incomplete payloads', () => {
    expect(metricDisplayValue('not-a-number', (value) => String(value))).toBe('—');
    expect(toHistogramSeries({
      bins: ['0-1'],
      counts: [3, 4],
      bin_edges: [0, 1, 2],
      min_length: 0,
      max_length: 2,
      mean_length: 1,
      median_length: 1,
    })).toEqual([
      { bin: '0-1', count: 3 },
      { bin: '1', count: 4 },
    ]);
    expect(parseZipfCurve(Array.from({ length: 250 }, (_, index) => ({ rank: index + 1, frequency: 1 })))).toHaveLength(200);
  });

  it('returns safe histogram series for malformed optional payloads', () => {
    expect(toHistogramSeries({ bins: ['0-1'], counts: 'invalid' })).toEqual([]);
    expect(toHistogramSeries({
      bins: ['0-1', null],
      counts: [3, '4', 'not-a-count', -1, Number.POSITIVE_INFINITY],
    })).toEqual([
      { bin: '0-1', count: 3 },
      { bin: '1', count: 4 },
    ]);
  });
});

describe('dashboardFileName', () => {
  it('removes path separators from dataset and tokenizer identifiers', () => {
    expect(dashboardFileName('dataset', 'custom/dataset_smoke')).toBe('dataset-custom-dataset_smoke-report.pdf');
    expect(dashboardFileName('tokenizer', 'owner/model', 'report-7')).toBe('tokenizer-owner-model-report-7.pdf');
  });

  it('falls back to a safe identifier when the input contains no filename characters', () => {
    expect(dashboardFileName('dataset', '///')).toBe('dataset-dashboard-report.pdf');
  });
});
