import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

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
  <div className="dataset-v2-card dataset-v2-chart-card">
    <div className="dataset-v2-card-header">
      <p className="panel-label">{title}</p>
    </div>
    {data.length === 0 ? (
      <div className="chart-placeholder">
        <p>{emptyMessage}</p>
      </div>
    ) : (
      <div className="dataset-v2-chart-body">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart
            data={data}
            margin={{ top: 8, right: 8, bottom: 0, left: 18 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#2d3440" />
            <XAxis dataKey="bin" hide />
            <YAxis
              stroke="#94a3b8"
              width={62}
              tick={{ fill: '#dbe5f1', fontSize: 12 }}
              axisLine={{ stroke: '#94a3b8' }}
              tickLine={{ stroke: '#94a3b8' }}
            />
            <Tooltip
              contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
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
