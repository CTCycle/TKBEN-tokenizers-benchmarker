export const CHART_COLORS = {
  axis: '#9ea7b3',
  axisTick: '#dbe5f1',
  grid: '#2d3440',
  tooltipBg: '#111827',
  tooltipBorder: '#374151',
  tooltipText: '#f8fafc',
  blue: '#4fc3f7',
  cyan: '#38bdf8',
  green: '#81c784',
  success: '#22c55e',
  yellow: '#facc15',
  amber: '#f59e0b',
  orange: '#ffb74d',
  red: '#f87171',
  rose: '#fb7185',
  pink: '#f06292',
  purple: '#ba68c8',
  violet: '#a78bfa',
  teal: '#4db6ac',
  slate: '#64748b',
} as const;

export const CHART_GRID_PROPS = {
  strokeDasharray: '3 3',
  stroke: CHART_COLORS.grid,
} as const;

export const CHART_AXIS_PROPS = {
  stroke: CHART_COLORS.axis,
} as const;

export const CHART_AXIS_TICK = {
  fill: CHART_COLORS.axisTick,
  fontSize: 12,
} as const;

export const CHART_TOOLTIP_STYLE = {
  backgroundColor: CHART_COLORS.tooltipBg,
  border: `1px solid ${CHART_COLORS.tooltipBorder}`,
} as const;

export const CHART_TOOLTIP_TEXT_STYLE = {
  color: CHART_COLORS.tooltipText,
} as const;

export const CHART_SERIES_COLORS = [
  CHART_COLORS.blue,
  CHART_COLORS.green,
  CHART_COLORS.orange,
  CHART_COLORS.pink,
  CHART_COLORS.purple,
  CHART_COLORS.teal,
] as const;

export const DATASET_DONUT_COLORS = [
  CHART_COLORS.amber,
  CHART_COLORS.rose,
  CHART_COLORS.cyan,
  '#34d399',
  CHART_COLORS.violet,
  '#f97316',
  CHART_COLORS.slate,
] as const;
