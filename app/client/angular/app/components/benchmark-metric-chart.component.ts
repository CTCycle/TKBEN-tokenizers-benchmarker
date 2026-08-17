import { Component, computed, input } from '@angular/core';
import type { BenchmarkDashboardWidgetData, BenchmarkVisualizationKind } from '../core/api/api.models';
import {
  type BenchmarkBucketView,
  bucketViews,
  classifyBenchmarkDataShape,
  distributionViews,
  formatBenchmarkAxisValue,
  formatBenchmarkTooltipValue,
  formatBenchmarkValue,
  histogramViews,
  pointViews,
  uniqueTokenizers,
} from '../core/utils/benchmark-dashboard-data';

@Component({
  selector: 'app-benchmark-metric-chart',
  templateUrl: './benchmark-metric-chart.component.html',
})
export class BenchmarkMetricChartComponent {
  readonly widget = input.required<BenchmarkDashboardWidgetData>();
  readonly visualization = input<BenchmarkVisualizationKind>('bar');
  protected readonly math = Math;
  protected readonly shape = computed(() => classifyBenchmarkDataShape(this.widget()));
  protected readonly points = computed(() => pointViews(this.widget()));
  protected readonly distributions = computed(() => distributionViews(this.widget()));
  protected readonly buckets = computed(() => bucketViews(this.widget()));
  protected readonly histogramBins = computed(() => histogramViews(this.widget()));
  protected readonly tokenizers = computed(() => uniqueTokenizers(this.widget()));
  protected readonly axisFractions = [0, 0.25, 0.5, 0.75, 1] as const;
  protected readonly histogramAxisFractions = [0, 0.5, 1] as const;
  protected readonly maxPointValue = computed(() => {
    const raw = Math.max(1, ...this.points().map((item) => Math.abs(item.value)));
    if (raw <= 1) return 1;
    const roughStep = raw / 4;
    const magnitude = 10 ** Math.floor(Math.log10(roughStep));
    const normalized = roughStep / magnitude;
    const step = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 2.2 ? 2.5 : normalized <= 3 ? 3 : normalized <= 5 ? 5 : 10;
    const intervalCount = Math.ceil(raw / (step * magnitude));
    const exactMultiple = Math.abs(raw / (step * magnitude) - intervalCount) < Number.EPSILON;
    return step * magnitude * (exactMultiple ? intervalCount + 1 : intervalCount);
  });
  protected readonly maxBucketValue = computed(() => Math.max(1, ...this.buckets().map((item) => Math.abs(item.value))));
  protected readonly maxHistogramCount = computed(() => Math.max(1, ...this.histogramBins().map((item) => item.count)));
  protected readonly boxScale = computed(() => {
    const values = this.distributions().flatMap((item) => [item.min, item.max]).filter(Number.isFinite);
    const min = values.length ? Math.min(...values) : 0;
    const max = values.length ? Math.max(...values) : 1;
    const logarithmic = min > 0 && max / min >= 50;
    const start = logarithmic ? min : Math.min(min, 0);
    return { min: start, max: Math.max(max, start + Number.EPSILON), logarithmic };
  });
  protected readonly histogramEdges = computed(() => {
    const edges = [...new Set(this.histogramBins().map((item) => item.bin_low))].sort((a, b) => a - b);
    const lastHigh = this.histogramBins().reduce((high, item) => Math.max(high, item.bin_high), 0);
    return [...edges, lastHigh];
  });
  protected readonly bucketsList = computed(() => [...new Set(this.buckets().map((item) => item.bucket))]);
  protected readonly bucketMinValue = computed(() => Math.min(0, ...this.buckets().map((item) => item.value)));
  protected readonly bucketMaxValue = computed(() => Math.max(1, ...this.buckets().map((item) => item.value)));

  protected readonly format = formatBenchmarkValue;
  protected readonly formatAxis = formatBenchmarkAxisValue;
  protected readonly formatTooltip = formatBenchmarkTooltipValue;

  protected pointX(index: number): number {
    return 108 + (index + 0.5) * (504 / Math.max(this.points().length, 1));
  }

  protected pointY(value: number): number {
    return 200 - (Math.abs(value) / this.maxPointValue()) * 190;
  }

  protected pointTickY(fraction: number): number { return 200 - fraction * 190; }
  protected pointTickValue(fraction: number): number { return this.maxPointValue() * fraction; }
  protected pointBarWidth(): number { return Math.min(56, Math.max(12, (504 / Math.max(this.points().length, 1)) * 0.68)); }

  protected horizontalY(index: number): number {
    return 46 + index * (140 / Math.max(this.points().length, 1));
  }

  protected horizontalWidth(value: number): number {
    return (Math.abs(value) / this.maxPointValue()) * 500;
  }

  protected forestScale(value: number): number {
    const points = this.points();
    const min = Math.min(0, ...points.map((item) => item.low));
    const max = Math.max(1, ...points.map((item) => item.high));
    return 112 + ((value - min) / Math.max(max - min, 1)) * 304;
  }

  protected boxScaleX(value: number): number {
    const scale = this.boxScale();
    const ratio = scale.logarithmic
      ? Math.log(Math.max(value, scale.min) / scale.min) / Math.log(scale.max / scale.min)
      : (value - scale.min) / Math.max(scale.max - scale.min, Number.EPSILON);
    return 116 + ratio * 288;
  }

  protected boxTickX(fraction: number): number { return 116 + fraction * 288; }
  protected boxTickValue(fraction: number): string {
    const scale = this.boxScale();
    const value = scale.logarithmic
      ? scale.min * (scale.max / scale.min) ** fraction
      : scale.min + (scale.max - scale.min) * fraction;
    return this.formatAxis(value, this.widget().display_format);
  }

  protected boxRowY(index: number): number {
    const rowHeight = (266 - 34) / Math.max(this.distributions().length, 1);
    return 34 + rowHeight * (index + 0.5);
  }

  protected boxHeight(): number {
    const rowHeight = (266 - 34) / Math.max(this.distributions().length, 1);
    return Math.max(6, Math.min(32, rowHeight * 0.58));
  }

  protected histogramX(index: number): number {
    return 24 + index * (272 / Math.max(this.histogramEdges().length - 1, 1));
  }

  protected histogramBarWidth(): number {
    return Math.max(2, 272 / Math.max(this.histogramEdges().length - 1, 1) - 1);
  }

  protected histogramHeight(count: number): number {
    return (count / this.maxHistogramCount()) * 134;
  }

  protected histogramTickY(fraction: number): number { return 160 - fraction * 134; }
  protected histogramTickValue(fraction: number): string { return this.formatAxis(this.maxHistogramCount() * fraction, 'number'); }

  protected bucketTickY(fraction: number): number { return 198 - fraction * 150; }
  protected bucketTickValue(fraction: number): string { return this.formatAxis(this.maxBucketValue() * fraction, this.widget().display_format); }
  protected bucketBarWidth(): number {
    const groupWidth = 540 / Math.max(this.bucketsList().length, 1);
    return Math.min(28, Math.max(6, groupWidth / Math.max(this.tokenizers().length, 1) - 4));
  }
  protected bucketGroupCenter(bucketIndex: number): number {
    return 50 + (bucketIndex + 0.5) * (540 / Math.max(this.bucketsList().length, 1));
  }

  protected bucketX(bucketIndex: number, tokenizerIndex: number): number {
    const bucketCount = Math.max(this.bucketsList().length, 1);
    const tokenizerCount = Math.max(this.tokenizers().length, 1);
    const groupWidth = 540 / bucketCount;
    const barWidth = this.bucketBarWidth();
    const totalWidth = tokenizerCount * barWidth + Math.max(0, tokenizerCount - 1) * 3;
    return 50 + bucketIndex * groupWidth + Math.max(0, (groupWidth - totalWidth) / 2) + tokenizerIndex * (barWidth + 3);
  }

  protected bucketY(value: number): number {
    return 198 - (Math.abs(value) / this.maxBucketValue()) * 150;
  }

  protected bucketValue(tokenizer: string, bucket: string): number | null {
    return this.buckets().find((item) => item.tokenizer === tokenizer && item.bucket === bucket)?.value ?? null;
  }

  protected bucketItem(tokenizer: string, bucket: string): BenchmarkBucketView | null {
    return this.buckets().find((item) => item.tokenizer === tokenizer && item.bucket === bucket) ?? null;
  }

  protected heatmapBackground(value: number): string {
    const range = Math.max(this.bucketMaxValue() - this.bucketMinValue(), 1);
    const intensity = Math.max(0, Math.min(1, (value - this.bucketMinValue()) / range));
    return `color-mix(in srgb, #4fc3f7 ${Math.round(20 + intensity * 70)}%, #111827)`;
  }

  protected tokenizerColor(tokenizer: string): string {
    return this.points().find((item) => item.tokenizer === tokenizer)?.color
      ?? this.distributions().find((item) => item.tokenizer === tokenizer)?.color
      ?? this.buckets().find((item) => item.tokenizer === tokenizer)?.color
      ?? this.histogramBins().find((item) => item.tokenizer === tokenizer)?.color
      ?? '#4fc3f7';
  }

  protected histogramBin(tokenizer: string, low: number): { count: number; proportion: number; bin_high: number } | null {
    const bin = this.histogramBins().find((item) => item.tokenizer === tokenizer && item.bin_low === low);
    return bin ? { count: bin.count, proportion: bin.proportion, bin_high: bin.bin_high } : null;
  }
}
