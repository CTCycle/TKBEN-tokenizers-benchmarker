import type { BenchmarkDashboardWidgetData } from '../../types/api';

export type BenchmarkDataShape = 'distribution' | 'bucket' | 'point';

export const classifyBenchmarkDataShape = (widget: BenchmarkDashboardWidgetData): BenchmarkDataShape =>
  widget.distributions.length || widget.histogram_bins.length
    ? 'distribution'
    : widget.buckets.length
      ? 'bucket'
      : 'point';
