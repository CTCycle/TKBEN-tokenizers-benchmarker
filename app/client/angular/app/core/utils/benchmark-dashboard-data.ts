import type {
  BenchmarkDashboardBucketPoint,
  BenchmarkDashboardDistribution,
  BenchmarkDashboardHistogramBin,
  BenchmarkDashboardPoint,
  BenchmarkDashboardWidgetData,
} from '../api/api.models';

export type BenchmarkDataShape = 'distribution' | 'bucket' | 'point';

export const classifyBenchmarkDataShape = (widget: BenchmarkDashboardWidgetData): BenchmarkDataShape =>
  widget.distributions.length || widget.histogram_bins.length
    ? 'distribution'
    : widget.buckets.length
      ? 'bucket'
      : 'point';

export const formatBenchmarkValue = (value: number, kind: string): string => {
  if (!Number.isFinite(value)) return 'N/A';
  if (kind === 'percent') return `${(value <= 1 ? value * 100 : value).toFixed(2)}%`;
  if (kind === 'milliseconds') return `${value.toLocaleString(undefined, { maximumFractionDigits: 3 })} ms`;
  if (kind === 'seconds') return `${value.toLocaleString(undefined, { maximumFractionDigits: 3 })} s`;
  if (kind === 'megabytes') return `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })} MB`;
  return value.toLocaleString(undefined, { maximumFractionDigits: 3 });
};

export const shortBenchmarkLabel = (value: string): string => value.length > 16 ? `${value.slice(0, 14)}…` : value;

const SERIES_COLORS = ['#4fc3f7', '#81c784', '#ffb74d', '#f06292', '#ba68c8', '#4db6ac'] as const;

export const benchmarkColorFor = (value: string): string =>
  SERIES_COLORS[[...value].reduce((sum, char) => sum + char.charCodeAt(0), 0) % SERIES_COLORS.length];

export interface BenchmarkPointView extends BenchmarkDashboardPoint {
  low: number;
  high: number;
  color: string;
  label: string;
}

export interface BenchmarkDistributionView extends BenchmarkDashboardDistribution {
  color: string;
  label: string;
}

export interface BenchmarkBucketView extends BenchmarkDashboardBucketPoint {
  color: string;
  label: string;
}

export interface BenchmarkHistogramView extends BenchmarkDashboardHistogramBin {
  color: string;
  label: string;
}

export const pointViews = (widget: BenchmarkDashboardWidgetData): BenchmarkPointView[] => widget.points.map((item) => ({
  ...item,
  low: item.interval_low ?? item.value,
  high: item.interval_high ?? item.value,
  color: benchmarkColorFor(item.tokenizer),
  label: shortBenchmarkLabel(item.tokenizer),
}));

export const distributionViews = (widget: BenchmarkDashboardWidgetData): BenchmarkDistributionView[] => widget.distributions.map((item) => ({
  ...item,
  color: benchmarkColorFor(item.tokenizer),
  label: shortBenchmarkLabel(item.tokenizer),
}));

export const bucketViews = (widget: BenchmarkDashboardWidgetData): BenchmarkBucketView[] => widget.buckets.map((item) => ({
  ...item,
  color: benchmarkColorFor(item.tokenizer),
  label: shortBenchmarkLabel(item.tokenizer),
}));

export const histogramViews = (widget: BenchmarkDashboardWidgetData): BenchmarkHistogramView[] => widget.histogram_bins.map((item) => ({
  ...item,
  color: benchmarkColorFor(item.tokenizer),
  label: shortBenchmarkLabel(item.tokenizer),
}));

export const uniqueTokenizers = (widget: BenchmarkDashboardWidgetData): string[] => [
  ...new Set([
    ...widget.points.map((item) => item.tokenizer),
    ...widget.distributions.map((item) => item.tokenizer),
    ...widget.buckets.map((item) => item.tokenizer),
    ...widget.histogram_bins.map((item) => item.tokenizer),
  ]),
];
