import type { BenchmarkDashboardWidgetData, BenchmarkVisualizationKind } from '../../types/api';
import { classifyBenchmarkDataShape } from '../../features/benchmark-dashboard/benchmarkDashboardChartUtils';
import { BenchmarkMetricChart } from './BenchmarkMetricCharts';
import { BenchmarkMetricDataTable } from './BenchmarkMetricDataTable';

type BenchmarkMetricWidgetProps = {
  widget: BenchmarkDashboardWidgetData;
  visualization: BenchmarkVisualizationKind;
  onVisualizationChange?: (visualization: BenchmarkVisualizationKind) => void;
};

const visualizationLabels: Record<BenchmarkVisualizationKind, string> = {
  bar: 'Vertical bar chart',
  horizontal_bar: 'Horizontal bar chart',
  interval_bar: 'Interval bar chart',
  dot_whisker: 'Forest plot',
  box_plot: 'Box plot',
  histogram: 'Histogram',
  grouped_bar: 'Grouped bar chart',
  heatmap: 'Heatmap',
};

export const BenchmarkMetricWidget = ({
  widget,
  visualization,
  onVisualizationChange,
}: BenchmarkMetricWidgetProps) => {
  const dataShape = classifyBenchmarkDataShape(widget);

  return (
    <article className={`benchmark-dashboard-widget benchmark-dashboard-widget--${widget.width}`} aria-label={`${widget.label} widget`}>
      <header>
        <div>
          <h3>{widget.label}</h3>
          <p className="cross-benchmark-chart-note">{widget.description} ({widget.unit})</p>
        </div>
        <div
          className="benchmark-visualization-switcher"
          role="group"
          aria-label={`${widget.label} visualization`}
          onPointerDown={(event) => event.stopPropagation()}
          onKeyDown={(event) => event.stopPropagation()}
        >
          {widget.compatible_visualizations.filter((choice) => choice !== 'heatmap').map((choice) => (
            <button
              key={choice}
              type="button"
              className="benchmark-visualization-button"
              aria-label={`Use ${visualizationLabels[choice]} for ${widget.label}`}
              title={visualizationLabels[choice]}
              aria-pressed={choice === visualization}
              onClick={() => onVisualizationChange?.(choice)}
            >
              {choice === 'bar' && '▮'}
              {choice === 'horizontal_bar' && '▰'}
              {choice === 'interval_bar' && '▥'}
              {choice === 'dot_whisker' && '⊙'}
              {choice === 'box_plot' && '▣'}
              {choice === 'histogram' && '▥'}
              {choice === 'grouped_bar' && '▤'}
            </button>
          ))}
        </div>
      </header>
      <div className={`benchmark-chart-stage benchmark-chart-stage--data-${dataShape} benchmark-chart-stage--${visualization}`}>
        <div className="benchmark-chart-stage__content">
          <BenchmarkMetricChart widget={widget} visualization={visualization} />
        </div>
      </div>
      <BenchmarkMetricDataTable widget={widget} />
    </article>
  );
};
