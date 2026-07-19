import { useCallback, useRef, useState } from 'react';
import { fetchAvailableDatasets } from '../services/datasetsApi';
import type { DatasetCatalogFilters, DatasetPreviewItem } from '../types/api';

type UseAvailableDatasetsResult = {
  availableDatasets: DatasetPreviewItem[];
  datasetsLoading: boolean;
  datasetsInitialized: boolean;
  refreshAvailableDatasets: (filters?: DatasetCatalogFilters) => Promise<DatasetPreviewItem[]>;
};

export const useAvailableDatasets = (): UseAvailableDatasetsResult => {
  const [availableDatasets, setAvailableDatasets] = useState<DatasetPreviewItem[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [datasetsInitialized, setDatasetsInitialized] = useState(false);
  const requestSequence = useRef(0);

  const refreshAvailableDatasets = useCallback(async (filters: DatasetCatalogFilters = {}) => {
    const requestId = ++requestSequence.current;
    setDatasetsLoading(true);
    try {
      const response = await fetchAvailableDatasets(filters);
      if (requestId === requestSequence.current) {
        setAvailableDatasets(response.datasets);
        setDatasetsInitialized(true);
      }
      return response.datasets;
    } finally {
      setDatasetsLoading(false);
    }
  }, []);

  return {
    availableDatasets,
    datasetsLoading,
    datasetsInitialized,
    refreshAvailableDatasets,
  };
};
