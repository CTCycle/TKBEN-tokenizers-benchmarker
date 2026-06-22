import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import {
  CHART_AXIS_PROPS,
  CHART_AXIS_TICK,
  CHART_GRID_PROPS,
  CHART_TOOLTIP_STYLE,
} from '../common/chartStyles';

type HistogramSeriesDatum = {
  bin: string;
  count: number;
};

type HistogramChartCardProps = {
  title: string;
  data: HistogramSeriesDatum[];
  emptyMessage: string;
  barFill: string;
  tooltipFormatter: (value: unknown) => [string, 'count'];
};

const HistogramChartCard = ({
  title,
  data,
  emptyMessage,
  barFill,
  tooltipFormatter,
}: HistogramChartCardProps) => (
  <div className="dataset-card dataset-chart-card">
    <div className="dataset-card-header">
      <p className="panel-label">{title}</p>
    </div>
    {data.length === 0 ? (
      <div className="chart-placeholder">
        <p>{emptyMessage}</p>
      </div>
    ) : (
      <div className="dataset-chart-body">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart
            data={data}
            margin={{ top: 8, right: 8, bottom: 0, left: 18 }}
          >
            <CartesianGrid {...CHART_GRID_PROPS} />
            <XAxis dataKey="bin" hide />
            <YAxis
              {...CHART_AXIS_PROPS}
              width={62}
              tick={CHART_AXIS_TICK}
              axisLine={CHART_AXIS_PROPS}
              tickLine={CHART_AXIS_PROPS}
            />
            <Tooltip
              contentStyle={CHART_TOOLTIP_STYLE}
              formatter={tooltipFormatter}
            />
            <Bar dataKey="count" fill={barFill} radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    )}
  </div>
);

export default HistogramChartCard;
