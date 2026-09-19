import type {
  BenchmarkDashboardBucketPoint,
  BenchmarkDashboardData,
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

const compactAxisNumber = (value: number): string => {
  const absolute = Math.abs(value);
  const suffix = absolute >= 1_000_000_000 ? 'B' : absolute >= 1_000_000 ? 'M' : absolute >= 10_000 ? 'k' : '';
  const divisor = suffix === 'B' ? 1_000_000_000 : suffix === 'M' ? 1_000_000 : suffix === 'k' ? 1_000 : 1;
  const scaled = value / divisor;
  const maximumFractionDigits = suffix ? (Math.abs(scaled) >= 100 ? 0 : Math.abs(scaled) >= 10 ? 1 : 2) : 3;
  return `${scaled.toLocaleString(undefined, { maximumFractionDigits })}${suffix}`;
};

export const formatBenchmarkAxisValue = (value: number, kind: string): string => {
  if (!Number.isFinite(value)) return 'N/A';
  if (kind === 'percent') return `${(value <= 1 ? value * 100 : value).toFixed(2)}%`;
  return compactAxisNumber(value);
};

export const formatBenchmarkTooltipValue = (value: number, kind: string, unit: string): string => {
  const formatted = formatBenchmarkValue(value, kind);
  return ['percent', 'milliseconds', 'seconds', 'megabytes'].includes(kind) ? formatted : `${formatted} ${unit}`;
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

export interface BenchmarkComparisonRow {
  tokenizer: string;
  deltaPercent: number | null;
  formattedDelta: string;
  isBaseline: boolean;
}

type BenchmarkDashboardWidgets = BenchmarkDashboardData | readonly BenchmarkDashboardWidgetData[];

const widgetsFromDashboard = (dashboard: BenchmarkDashboardWidgets): readonly BenchmarkDashboardWidgetData[] =>
  Array.isArray(dashboard) ? dashboard : (dashboard as BenchmarkDashboardData).widgets;

export const benchmarkPointValue = (
  widget: BenchmarkDashboardWidgetData,
  tokenizer: string,
): number | null => {
  const point = (widget.points ?? []).find((item) => item.tokenizer === tokenizer);
  return point && Number.isFinite(point.value) ? point.value : null;
};

export const benchmarkDistributionMedian = (
  widget: BenchmarkDashboardWidgetData,
  tokenizer: string,
): number | null => {
  const distribution = (widget.distributions ?? []).find((item) => item.tokenizer === tokenizer);
  return distribution && Number.isFinite(distribution.median) ? distribution.median : null;
};

export const benchmarkComparisonTokenizers = (dashboard: BenchmarkDashboardWidgets): string[] => [
  ...new Set(
    widgetsFromDashboard(dashboard).flatMap((widget) => [
      ...(widget.points ?? []).filter((item) => Number.isFinite(item.value)).map((item) => item.tokenizer),
      ...(widget.distributions ?? []).filter((item) => Number.isFinite(item.median)).map((item) => item.tokenizer),
    ]),
  ),
];

export const relativeBenchmarkDelta = (
  candidateValue: number | null | undefined,
  baselineValue: number | null | undefined,
): number | null => {
  if (
    candidateValue === null || candidateValue === undefined
    || baselineValue === null || baselineValue === undefined
    || !Number.isFinite(candidateValue)
    || !Number.isFinite(baselineValue)
    || baselineValue === 0
  ) return null;
  const delta = ((candidateValue - baselineValue) / baselineValue) * 100;
  return Number.isFinite(delta) ? delta : null;
};

export const formatBenchmarkDelta = (deltaPercent: number | null, isBaseline = false): string => {
  if (isBaseline) return 'Baseline';
  if (deltaPercent === null || !Number.isFinite(deltaPercent)) return 'N/A';
  return `${deltaPercent > 0 ? '+' : ''}${deltaPercent.toFixed(2)}%`;
};

export const benchmarkComparisonRows = (
  widget: BenchmarkDashboardWidgetData,
  baselineTokenizer: string | null | undefined,
): BenchmarkComparisonRow[] => {
  if (!baselineTokenizer) return [];

  if ((widget.distributions ?? []).length) {
    const baselineValue = benchmarkDistributionMedian(widget, baselineTokenizer);
    return (widget.distributions ?? []).map((item) => {
      const isBaseline = item.tokenizer === baselineTokenizer;
      const deltaPercent = isBaseline ? null : relativeBenchmarkDelta(item.median, baselineValue);
      return {
        tokenizer: item.tokenizer,
        deltaPercent,
        formattedDelta: formatBenchmarkDelta(deltaPercent, isBaseline),
        isBaseline,
      };
    });
  }

  if ((widget.points ?? []).length) {
    const baselineValue = benchmarkPointValue(widget, baselineTokenizer);
    return (widget.points ?? []).map((item) => {
      const isBaseline = item.tokenizer === baselineTokenizer;
      const deltaPercent = isBaseline ? null : relativeBenchmarkDelta(item.value, baselineValue);
      return {
        tokenizer: item.tokenizer,
        deltaPercent,
        formattedDelta: formatBenchmarkDelta(deltaPercent, isBaseline),
        isBaseline,
      };
    });
  }

  return [];
};
