import { Component, computed, input } from '@angular/core';
import type { BenchmarkDashboardWidgetData, BenchmarkVisualizationKind } from '../core/api/api.models';
import {
  type BenchmarkBucketView,
  bucketViews,
  classifyBenchmarkDataShape,
  distributionViews,
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
  protected readonly maxPointValue = computed(() => Math.max(1, ...this.points().map((item) => Math.abs(item.value))));
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

  protected readonly format = formatBenchmarkValue;

  protected pointX(index: number): number {
    return 80 + index * (500 / Math.max(this.points().length - 1, 1));
  }

  protected pointY(value: number): number {
    return 198 - (Math.abs(value) / this.maxPointValue()) * 156;
  }

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

  protected bucketX(bucketIndex: number, tokenizerIndex: number): number {
    const bucketCount = Math.max(this.bucketsList().length, 1);
    const tokenizerCount = Math.max(this.tokenizers().length, 1);
    const groupWidth = 540 / bucketCount;
    const barWidth = Math.min(28, Math.max(6, groupWidth / tokenizerCount - 4));
    return 50 + bucketIndex * groupWidth + tokenizerIndex * (barWidth + 3);
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
