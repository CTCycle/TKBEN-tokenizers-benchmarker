import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { BenchmarkDashboardWidgetData, BenchmarkVisualizationKind } from '../../types/api';
import { CHART_AXIS_PROPS, CHART_AXIS_TICK, CHART_COLORS, CHART_GRID_PROPS, CHART_SERIES_COLORS, CHART_TOOLTIP_STYLE } from '../../common/chartStyles';
import { formatBenchmarkValue } from '../../features/benchmark-dashboard/benchmarkDashboardFormatters';
import { useCompactChart } from '../../hooks/useCompactChart';

export type BenchmarkMetricChartProps = {
  widget: BenchmarkDashboardWidgetData;
  visualization: BenchmarkVisualizationKind;
};

type BenchmarkMetricWidgetOnlyProps = {
  widget: BenchmarkDashboardWidgetData;
};

const shortLabel = (value: string): string => value.length > 16 ? `${value.slice(0, 14)}…` : value;

const colorFor = (value: string): string =>
  CHART_SERIES_COLORS[[...value].reduce((sum, char) => sum + char.charCodeAt(0), 0) % CHART_SERIES_COLORS.length];

const chartMargin = { top: 10, right: 18, left: 8, bottom: 30 };

const referenceTicks = (min: number, max: number, logarithmic: boolean): Array<{ ratio: number; value: number }> =>
  Array.from({ length: 6 }, (_, index) => {
    const ratio = index / 5;
    const value = logarithmic
      ? min * Math.pow(max / min, ratio)
      : min + (max - min) * ratio;
    return { ratio, value };
  });

type ChartTooltipProps = {
  active?: boolean;
  payload?: Array<{
    name?: string;
    value?: number;
    color?: string;
    payload?: { fullName?: string; unit?: string };
  }>;
  label?: string;
  format: string;
};

const ChartTooltip = ({ active, payload, label, format }: ChartTooltipProps) => {
  const [firstItem] = payload ?? [];
  if (!active || !firstItem) return null;
  return (
    <div className="benchmark-chart-tooltip">
      <strong>{firstItem.payload?.fullName ?? label}</strong>
      {payload?.map((item) => (
        <span key={item.name ?? String(item.value)} style={{ color: item.color }}>
          {item.name ? `${item.name}: ` : ''}
          {formatBenchmarkValue(Number(item.value), format)}
        </span>
      ))}
    </div>
  );
};

const RechartsMetric = ({ widget, visualization }: BenchmarkMetricChartProps) => {
  const data = widget.points.map((item) => ({
    name: shortLabel(item.tokenizer),
    fullName: item.tokenizer,
    value: item.value,
    error: item.interval_low === null || item.interval_high === null
      ? [0, 0]
      : [item.value - item.interval_low, item.interval_high - item.value],
    color: colorFor(item.tokenizer),
  }));
  const horizontal = visualization === 'horizontal_bar';
  const interval = visualization === 'interval_bar';

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={data}
        layout={horizontal ? 'vertical' : 'horizontal'}
        barCategoryGap="26%"
        margin={chartMargin}
      >
        <CartesianGrid {...CHART_GRID_PROPS} />
        <XAxis
          dataKey={horizontal ? 'value' : 'name'}
          type={horizontal ? 'number' : 'category'}
          tick={CHART_AXIS_TICK}
          axisLine={CHART_AXIS_PROPS}
          label={{
            value: horizontal ? widget.unit : undefined,
            position: 'insideBottom',
            offset: -18,
            fill: CHART_COLORS.axisTick,
          }}
        />
        <YAxis
          dataKey={horizontal ? 'name' : 'value'}
          type={horizontal ? 'category' : 'number'}
          tick={CHART_AXIS_TICK}
          axisLine={CHART_AXIS_PROPS}
          label={{
            value: horizontal ? undefined : widget.unit,
            angle: -90,
            position: 'insideLeft',
            fill: CHART_COLORS.axisTick,
          }}
        />
        <Tooltip content={<ChartTooltip format={widget.display_format} />} contentStyle={CHART_TOOLTIP_STYLE} />
        <Bar
          dataKey="value"
          name={widget.unit}
          maxBarSize={horizontal ? 30 : 44}
          radius={horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0]}
        >
          {data.map((item) => <Cell key={item.fullName} fill={item.color} />)}
          {interval && <ErrorBar dataKey="error" width={5} stroke={CHART_COLORS.yellow} />}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

const ForestPlot = ({ widget }: BenchmarkMetricWidgetOnlyProps) => {
  const values = widget.points.flatMap((item) => [
    item.interval_low ?? item.value,
    item.interval_high ?? item.value,
    item.value,
  ]);
  const compact = useCompactChart();
  const widePlot = widget.width === 'wide' && !compact;
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 1);
  const viewBoxWidth = widePlot ? 1600 : 440;
  const plotStart = widePlot ? 400 : 112;
  const plotEnd = widePlot ? 1560 : 416;
  const scale = (value: number): number =>
    plotStart + ((value - min) / Math.max(max - min, 1)) * (plotEnd - plotStart);
  const rowHeight = 72;
  const chartTop = 54;
  const height = Math.max(280, widget.points.length * rowHeight + 68);

  return (
    <svg
      className="benchmark-custom-chart"
      viewBox={`0 0 ${viewBoxWidth} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={`${widget.label} forest plot in ${widget.unit}`}
    >
      <text x={plotStart} y="20" fill={CHART_COLORS.axisTick} fontSize="12">{widget.unit}</text>
      {widget.points.map((item, index) => {
        const y = chartTop + index * rowHeight;
        const low = item.interval_low ?? item.value;
        const high = item.interval_high ?? item.value;
        return (
          <g key={item.tokenizer}>
            <title>{`${item.tokenizer}: ${formatBenchmarkValue(item.value, widget.display_format)}; interval ${formatBenchmarkValue(low, widget.display_format)}–${formatBenchmarkValue(high, widget.display_format)}`}</title>
            <text x={plotStart - 12} y={y + 5} fill={CHART_COLORS.axisTick} textAnchor="end" fontSize="13">{shortLabel(item.tokenizer)}</text>
            <line x1={scale(low)} x2={scale(high)} y1={y} y2={y} stroke={CHART_COLORS.yellow} strokeWidth="4" />
            <line x1={scale(low)} x2={scale(low)} y1={y - 10} y2={y + 10} stroke={CHART_COLORS.yellow} strokeWidth="2" />
            <line x1={scale(high)} x2={scale(high)} y1={y - 10} y2={y + 10} stroke={CHART_COLORS.yellow} strokeWidth="2" />
            <circle cx={scale(item.value)} cy={y} r="8" fill={colorFor(item.tokenizer)} />
          </g>
        );
      })}
    </svg>
  );
};

const BoxPlot = ({ widget }: BenchmarkMetricWidgetOnlyProps) => {
  const compact = useCompactChart();
  const widePlot = widget.width === 'wide' && !compact;
  const values = widget.distributions
    .flatMap((item) => [item.min, item.max])
    .filter(Number.isFinite);
  const dataMin = Math.min(...values);
  const dataMax = Math.max(...values);
  const logarithmic = dataMin > 0 && dataMax / dataMin >= 50;
  const min = logarithmic ? dataMin : Math.min(dataMin, 0);
  const max = Math.max(dataMax, min + Number.EPSILON);
  const scaleValue = (value: number): number => logarithmic
    ? Math.log(value / min) / Math.log(max / min)
    : (value - min) / Math.max(max - min, Number.EPSILON);
  const viewBoxWidth = widePlot ? 1600 : 440;
  const viewBoxHeight = widePlot ? 240 : 300;
  const plotStart = widePlot ? 320 : 116;
  const plotEnd = widePlot ? 1280 : 404;
  const axisY = viewBoxHeight - 34;
  const rowHeight = (axisY - 34) / Math.max(widget.distributions.length, 1);
  const plotHeight = Math.max(6, Math.min(32, rowHeight * 0.58));
  const scale = (value: number): number => plotStart + scaleValue(value) * (plotEnd - plotStart);
  const ticks = referenceTicks(min, max, logarithmic);

  return (
    <svg
      className="benchmark-custom-chart"
      viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={`${widget.label} box plot in ${widget.unit}${logarithmic ? ', log scale' : ''}`}
    >
      <text x={plotStart} y="18" fill={CHART_COLORS.axisTick} fontSize="12">
        {logarithmic ? `Log scale · ${widget.unit}` : widget.unit}
      </text>
      {widget.distributions.map((item, index) => {
        const y = 34 + rowHeight * (index + 0.5);
        const q1 = scale(item.q1);
        const q3 = scale(item.q3);
        const boxWidth = Math.max(q3 - q1, 14);
        const boxX = Math.min(q1, plotEnd - boxWidth);
        return (
          <g key={item.tokenizer}>
            <title>{`${item.tokenizer}: min ${formatBenchmarkValue(item.min, widget.display_format)}, Q1 ${formatBenchmarkValue(item.q1, widget.display_format)}, median ${formatBenchmarkValue(item.median, widget.display_format)}, Q3 ${formatBenchmarkValue(item.q3, widget.display_format)}, max ${formatBenchmarkValue(item.max, widget.display_format)}`}</title>
            <text x={plotStart - 12} y={y + 5} fill={CHART_COLORS.axisTick} textAnchor="end" fontSize="13">{shortLabel(item.tokenizer)}</text>
            <line x1={scale(item.min)} x2={scale(item.max)} y1={y} y2={y} stroke={CHART_COLORS.axis} strokeWidth="3" />
            <line x1={scale(item.min)} x2={scale(item.min)} y1={y - plotHeight / 2 - 2} y2={y + plotHeight / 2 + 2} stroke={CHART_COLORS.axis} strokeWidth="2" />
            <line x1={scale(item.max)} x2={scale(item.max)} y1={y - plotHeight / 2 - 2} y2={y + plotHeight / 2 + 2} stroke={CHART_COLORS.axis} strokeWidth="2" />
            <rect x={boxX} y={y - plotHeight / 2} width={boxWidth} height={plotHeight} rx="5" fill={colorFor(item.tokenizer)} fillOpacity="0.78" />
            <line x1={scale(item.median)} x2={scale(item.median)} y1={y - plotHeight / 2 - 2} y2={y + plotHeight / 2 + 2} stroke={CHART_COLORS.yellow} strokeWidth="4" />
          </g>
        );
      })}
      <g aria-hidden="true">
        <line x1={plotStart} x2={plotEnd} y1={axisY} y2={axisY} stroke={CHART_COLORS.axis} strokeWidth="2" />
        {ticks.map(({ ratio, value }, index) => {
          const x = plotStart + ratio * (plotEnd - plotStart);
          return (
            <g key={`${value}-${index}`}>
              <line x1={x} x2={x} y1={axisY} y2={axisY + 7} stroke={CHART_COLORS.axis} />
              <text x={x} y={axisY + 19} textAnchor="middle" fill={CHART_COLORS.axisTick} fontSize="11">{formatBenchmarkValue(value, widget.display_format)}</text>
            </g>
          );
        })}
        <text x={(plotStart + plotEnd) / 2} y={viewBoxHeight - 4} textAnchor="middle" fill={CHART_COLORS.axisTick} fontSize="11">{widget.unit}</text>
      </g>
    </svg>
  );
};

const HistogramSmallMultiples = ({ widget }: BenchmarkMetricWidgetOnlyProps) => {
  const compact = useCompactChart();
  const widePlot = widget.width === 'wide' && !compact;
  const tokenizers = [...new Set(widget.histogram_bins.map((item) => item.tokenizer))];
  const edges = [...new Set(widget.histogram_bins.map((item) => item.bin_low))].sort((a, b) => a - b);
  const lastHigh = widget.histogram_bins.reduce((high, item) => Math.max(high, item.bin_high), 0);
  const bins = [...edges, lastHigh];
  const maxCount = Math.max(...widget.histogram_bins.map((item) => item.count), 1);
  const viewBoxWidth = widePlot ? 600 : 320;
  const plotStart = widePlot ? 45 : 24;
  const plotEnd = widePlot ? 555 : 296;
  const plotWidth = plotEnd - plotStart;

  return (
    <div className="benchmark-histogram-grid" aria-label={`${widget.label} histogram small multiples`}>
      {tokenizers.map((tokenizer) => (
        <div className="benchmark-histogram-small-multiple" key={tokenizer}>
          <strong title={tokenizer}>{shortLabel(tokenizer)}</strong>
          <svg className="benchmark-custom-chart" viewBox={`0 0 ${viewBoxWidth} 190`} preserveAspectRatio="xMidYMid meet" role="img" aria-label={`${widget.label} for ${tokenizer} in ${widget.unit}`}>
            <title>{`${tokenizer} histogram · ${widget.unit}`}</title>
            {bins.slice(0, -1).map((low, index) => {
              const high = bins[index + 1];
              const bin = widget.histogram_bins.find((item) => item.tokenizer === tokenizer && item.bin_low === low);
              const count = bin?.count ?? 0;
              const x = plotStart + index * (plotWidth / Math.max(bins.length - 1, 1));
              const width = Math.max(2, plotWidth / Math.max(bins.length - 1, 1) - 1);
              const height = (count / maxCount) * 134;
              return (
                <rect key={`${tokenizer}-${low}`} x={x} y={160 - height} width={width} height={height} fill={colorFor(tokenizer)}>
                  <title>{`${tokenizer}: ${low}–${high}, ${count} observations (${((bin?.proportion ?? 0) * 100).toFixed(1)}%)`}</title>
                </rect>
              );
            })}
            <line x1={plotStart} x2={plotEnd} y1="160" y2="160" stroke={CHART_COLORS.axis} />
            <text x={plotStart} y="178" fill={CHART_COLORS.axisTick} fontSize="10">{formatBenchmarkValue(bins[0] ?? 0, widget.display_format)}</text>
            <text x={plotEnd} y="178" textAnchor="end" fill={CHART_COLORS.axisTick} fontSize="10">{formatBenchmarkValue(bins.at(-1) ?? 0, widget.display_format)}</text>
          </svg>
        </div>
      ))}
    </div>
  );
};

const GroupedBar = ({ widget }: BenchmarkMetricWidgetOnlyProps) => {
  const buckets = [...new Set(widget.buckets.map((item) => item.bucket))];
  const tokenizers = [...new Set(widget.buckets.map((item) => item.tokenizer))];
  const data = buckets.map((bucket) => Object.fromEntries([
    ['name', shortLabel(bucket)],
    ['fullName', bucket],
    ...tokenizers.map((tokenizer) => [
      tokenizer,
      widget.buckets.find((item) => item.bucket === bucket && item.tokenizer === tokenizer)?.value ?? null,
    ]),
  ])) as Array<Record<string, string | number | null>>;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={chartMargin}>
        <CartesianGrid {...CHART_GRID_PROPS} />
        <XAxis dataKey="name" tick={CHART_AXIS_TICK} axisLine={CHART_AXIS_PROPS} />
        <YAxis tick={CHART_AXIS_TICK} axisLine={CHART_AXIS_PROPS} label={{ value: widget.unit, angle: -90, position: 'insideLeft', fill: CHART_COLORS.axisTick }} />
        <Tooltip content={<ChartTooltip format={widget.display_format} />} contentStyle={CHART_TOOLTIP_STYLE} />
        {tokenizers.map((tokenizer) => <Bar key={tokenizer} dataKey={tokenizer} name={tokenizer} fill={colorFor(tokenizer)} maxBarSize={28} radius={[3, 3, 0, 0]} />)}
      </BarChart>
    </ResponsiveContainer>
  );
};

const Heatmap = ({ widget }: BenchmarkMetricWidgetOnlyProps) => {
  const tokenizers = [...new Set(widget.buckets.map((item) => item.tokenizer))];
  const buckets = [...new Set(widget.buckets.map((item) => item.bucket))];
  const byKey = new Map(widget.buckets.map((item) => [`${item.tokenizer}-${item.bucket}`, item.value]));
  const values = widget.buckets.map((item) => item.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const intensity = (value: number): number => (value - min) / Math.max(max - min, 1);

  return (
    <div className="benchmark-heatmap-wrap">
      <div className="benchmark-heatmap" role="grid" aria-label={`${widget.label} heatmap`} style={{ gridTemplateColumns: `minmax(0, 1.2fr) repeat(${buckets.length}, minmax(0, 1fr))` }}>
        <span className="benchmark-heatmap-corner">Tokenizer / bucket</span>
        {buckets.map((bucket) => <strong key={bucket}>{shortLabel(bucket)}</strong>)}
        {tokenizers.map((tokenizer) => (
          <span className="benchmark-heatmap-row" role="row" key={`${tokenizer}-row`}>
            <strong>{shortLabel(tokenizer)}</strong>
            {buckets.map((bucket) => {
              const value = byKey.get(`${tokenizer}-${bucket}`);
              return (
                <span
                  key={`${tokenizer}-${bucket}`}
                  className="benchmark-heatmap-cell"
                  role="gridcell"
                  tabIndex={0}
                  style={{ backgroundColor: value === undefined ? CHART_COLORS.tooltipBg : `color-mix(in srgb, ${CHART_COLORS.blue} ${Math.round(20 + intensity(value) * 70)}%, ${CHART_COLORS.tooltipBg})` }}
                  aria-label={`${tokenizer}, ${bucket}: ${value === undefined ? 'not available' : `${formatBenchmarkValue(value, widget.display_format)} ${widget.unit}`}`}
                >
                  {value === undefined ? '—' : formatBenchmarkValue(value, widget.display_format)}
                </span>
              );
            })}
          </span>
        ))}
      </div>
      <div className="benchmark-heatmap-legend" aria-label={`Heatmap scale from ${formatBenchmarkValue(min, widget.display_format)} to ${formatBenchmarkValue(max, widget.display_format)}`}>
        <span>Low</span><i /><span>High</span>
        <small>{formatBenchmarkValue(min, widget.display_format)}–${formatBenchmarkValue(max, widget.display_format)} {widget.unit}</small>
      </div>
    </div>
  );
};

export const BenchmarkMetricChart = ({ widget, visualization }: BenchmarkMetricChartProps) => {
  if (!widget.points.length && !widget.distributions.length && !widget.buckets.length) {
    return <p className="benchmark-dashboard-empty">No metric data available.</p>;
  }
  if (visualization === 'box_plot') return <BoxPlot widget={widget} />;
  if (visualization === 'histogram') return <HistogramSmallMultiples widget={widget} />;
  if (visualization === 'dot_whisker') return <ForestPlot widget={widget} />;
  if (visualization === 'grouped_bar') return <GroupedBar widget={widget} />;
  if (visualization === 'heatmap') return <Heatmap widget={widget} />;
  return <RechartsMetric widget={widget} visualization={visualization} />;
};
