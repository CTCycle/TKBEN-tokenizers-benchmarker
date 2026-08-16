import type { HistogramData, WordCloudTerm, WordFrequency } from '../api/api.models';

export const toNumber = (value: unknown, fallback = 0): number => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
};

export const normalizePercent = (value: number): string => `${(value * 100).toFixed(2)}%`;

export const normalizeCount = (value: number): string => Math.round(value).toLocaleString();

export const hasMetricValue = (value: unknown): boolean => {
  if (typeof value === 'number') {
    return Number.isFinite(value);
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) {
      return false;
    }
    return Number.isFinite(Number(trimmed));
  }
  return false;
};

export const metricDisplayValue = (
  value: unknown,
  formatter: (numeric: number) => string,
): string => (
  hasMetricValue(value)
    ? formatter(toNumber(value))
    : '—'
);

export const toHistogramSeries = (histogram: HistogramData | null): { bin: string; count: number }[] => {
  if (!histogram) {
    return [];
  }
  return histogram.counts.map((count, index) => ({
    bin: histogram.bins[index] ?? String(index),
    count,
  }));
};

export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export const parseWordFrequencyItems = (value: unknown): WordFrequency[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => {
      if (!isRecord(item)) {
        return null;
      }
      const word = typeof item['word'] === 'string' ? item['word'] : '';
      if (!word) {
        return null;
      }
      const count = Math.max(0, Math.round(toNumber(item['count'], 0)));
      return { word, count };
    })
    .filter((item): item is WordFrequency => item !== null && item.count > 0);
};

export const parseWordCloudTerms = (value: unknown): WordCloudTerm[] => {
  if (!Array.isArray(value)) {
    return [];
  }

  const terms = value
    .map((item) => {
      if (!isRecord(item)) {
        return null;
      }
      const word = typeof item['word'] === 'string' ? item['word'] : '';
      if (!word) {
        return null;
      }
      return {
        word,
        count: Math.max(0, Math.round(toNumber(item['count'], 0))),
        weight: toNumber(item['weight'], 0),
      };
    })
    .filter((item): item is WordCloudTerm => item !== null && item.count > 0);

  if (!terms.length) {
    return [];
  }

  const maxCount = Math.max(...terms.map((item) => item.count));
  return terms.map((item) => ({
    ...item,
    weight: item.weight > 0
      ? item.weight
      : Math.max(1, Math.round((item.count / Math.max(1, maxCount)) * 100)),
  }));
};

export const parseZipfCurve = (value: unknown): { rank: number; frequency: number }[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item, index) => {
      if (!isRecord(item)) {
        return null;
      }
      return {
        rank: toNumber(item['rank'], index + 1),
        frequency: toNumber(item['frequency'], 0),
      };
    })
    .filter((item): item is { rank: number; frequency: number } => item !== null && item.rank > 0 && item.frequency > 0)
    .sort((a, b) => a.rank - b.rank)
    .slice(0, 200);
};

export const tooltipPercentFormatter = (value: unknown): string =>
  normalizePercent(toNumber(value, 0));

export const tooltipCountFormatter = (value: unknown): [string, 'count'] => [
  normalizeCount(toNumber(value, 0)),
  'count',
];

export const buildZipfCurveFromWordFrequencies = (
  items: WordFrequency[],
): { rank: number; frequency: number }[] =>
  items
    .filter((item) => item.count > 0)
    .sort((a, b) => b.count - a.count || a.word.localeCompare(b.word))
    .map((item, index) => ({
      rank: index + 1,
      frequency: item.count,
    }))
    .slice(0, 200);

export const buildWordCloudFromWordFrequencies = (items: WordFrequency[]): WordCloudTerm[] => {
  const ranked = items
    .filter((item) => item.count > 0)
    .sort((a, b) => b.count - a.count || a.word.localeCompare(b.word))
    .slice(0, 120);
  if (!ranked.length) {
    return [];
  }
  const maxCount = Math.max(...ranked.map((item) => item.count));
  return ranked.map((item) => ({
    word: item.word,
    count: item.count,
    weight: Math.max(1, Math.round((item.count / Math.max(1, maxCount)) * 100)),
  }));
};
