export const formatBenchmarkValue = (value: number, kind: string): string => {
  if (!Number.isFinite(value)) return 'N/A';
  if (kind === 'percent') return `${(value <= 1 ? value * 100 : value).toFixed(2)}%`;
  if (kind === 'milliseconds') return `${value.toLocaleString(undefined, { maximumFractionDigits: 3 })} ms`;
  if (kind === 'seconds') return `${value.toLocaleString(undefined, { maximumFractionDigits: 3 })} s`;
  if (kind === 'megabytes') return `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })} MB`;
  return value.toLocaleString(undefined, { maximumFractionDigits: 3 });
};
