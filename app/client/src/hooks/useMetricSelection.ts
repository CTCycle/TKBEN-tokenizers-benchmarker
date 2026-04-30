import { useCallback, useMemo, useState } from 'react';

export interface MetricSelectionMetric {
  key: string;
  label: string;
}

export interface MetricSelectionCategory {
  category_key: string;
  category_label: string;
  metrics: MetricSelectionMetric[];
}

export function useMetricSelection(categories: MetricSelectionCategory[]) {
  const [selectedMetricKeys, setSelectedMetricKeys] = useState<string[]>([]);

  const allMetricKeys = useMemo(
    () => categories.flatMap((category) => category.metrics.map((metric) => metric.key)),
    [categories],
  );

  const resetSelectionToAll = useCallback(() => {
    setSelectedMetricKeys(allMetricKeys);
  }, [allMetricKeys]);

  const ensureSelectionInitialized = useCallback(() => {
    if (allMetricKeys.length === 0) {
      return;
    }
    setSelectedMetricKeys((current) => (current.length > 0 ? current : allMetricKeys));
  }, [allMetricKeys]);

  const toggleMetric = useCallback((metricKey: string, enabled: boolean) => {
    setSelectedMetricKeys((current) => {
      if (enabled) {
        if (current.includes(metricKey)) {
          return current;
        }
        return [...current, metricKey];
      }
      return current.filter((key) => key !== metricKey);
    });
  }, []);

  const toggleCategoryByKeys = useCallback((metricKeys: string[], enabled: boolean) => {
    setSelectedMetricKeys((current) => {
      if (enabled) {
        return Array.from(new Set([...current, ...metricKeys]));
      }
      return current.filter((key) => !metricKeys.includes(key));
    });
  }, []);

  return {
    allMetricKeys,
    selectedMetricKeys,
    toggleMetric,
    toggleCategoryByKeys,
    resetSelectionToAll,
    ensureSelectionInitialized,
  };
}
